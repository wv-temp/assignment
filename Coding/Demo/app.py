

# ------------------------------
# Auto .env load
# ------------------------------
from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# Imports & Config
# ------------------------------
import os
import io
import re
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

from PIL import Image
import dateparser
import gradio as gr
from random import randint, choice
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

# Database (SQLAlchemy + pgvector)
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector

# FastMCP
from fastmcp import FastMCP

from transformers import AutoTokenizer, AutoModelForCausalLM, DonutProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer
import uvicorn

DATABASE_URL = os.environ.get(
    "DATABASE_URL"
)
DONUT_MODEL_NAME = os.environ.get("DONUT_MODEL")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL")
EMB_DIM = int(os.environ.get("EMB_DIM"))

# =============================================
# Models & DB 
# =============================================
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class ReceiptItem(Base):

    __tablename__ = "receipt_items"
    id = Column(Integer, primary_key=True, index=True)
    item = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    dt = Column(DateTime, nullable=False)
    restaurant = Column(String, nullable=True)
    embedding = Column(Vector(EMB_DIM), nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


# =============================================
# Embedding & Models Loading
# =============================================

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_text(text: str) -> List[float]:
    model = get_embedding_model()
    vec = model.encode(text)

    try:
        py = vec.tolist()
    except Exception:
        py = list(vec)
    
    return [float(x) for x in py]

_donut_processor = None
_donut_model = None
_donut_device = None
_qwen_tokenizer = None
_qwen_model = None


def load_donut_if_needed():
    global _donut_processor, _donut_model, _donut_device
    if _donut_model is not None:
        return

    _donut_processor = DonutProcessor.from_pretrained(DONUT_MODEL_NAME)
    _donut_model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_NAME)
    _donut_device = "cuda" if torch.cuda.is_available() else "cpu"
    _donut_model.to(_donut_device)

def load_qwen_if_needed():
    global _qwen_tokenizer, _qwen_model
    if _qwen_model is not None:
        return

    model_name = "Qwen/Qwen3-0.6B"

    _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )


def parse_with_donut(image: Image.Image) -> Dict[str, Any]:
    load_donut_if_needed()

    pixel_values = _donut_processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = _donut_processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    outputs = _donut_model.generate(
        pixel_values.to(_donut_device),
        decoder_input_ids=decoder_input_ids.to(_donut_device),
        max_length=_donut_model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=_donut_processor.tokenizer.pad_token_id,
        eos_token_id=_donut_processor.tokenizer.eos_token_id,
        num_beams=1,
        use_cache=True,
        bad_words_ids=[[ _donut_processor.tokenizer.unk_token_id ]],
        return_dict_in_generate=True,
    )

    sequence = _donut_processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(_donut_processor.tokenizer.eos_token, "")
    sequence = sequence.replace(_donut_processor.tokenizer.pad_token, "")
    sequence = re.sub(r"^<.*?>", "", sequence).strip()

    parsed_json = _donut_processor.token2json(sequence)

    vendor = parsed_json.get("vendor") or parsed_json.get("merchant") or parsed_json.get("store") or "Unknown"

    dt = None
    for k in ("date", "invoice_date", "document_date", "purchase_date"):
        if k in parsed_json:
            dt = dateparser.parse(parsed_json[k], settings={"PREFER_DATES_FROM": "past"})
            if dt:
                break
    if dt is None:
        dt = datetime.now()

    items = []
    line_items = parsed_json.get("line_items") or parsed_json.get("items") or []
    for li in line_items:
        desc = li.get("description") or li.get("name") or li.get("item") or "Unknown"
        price_raw = (
            li.get("total")
            or li.get("price")
            or li.get("amount")
            or li.get("unit_price")
        )
        price = 0.0
        if price_raw:
            try:
                price = float(str(price_raw).replace(",", "."))
            except Exception:
                price = 0.0
        items.append({"item": desc, "price": price})

    if items and all(i["price"] == 0.0 for i in items):
        inv_raw = parsed_json.get("invoice_total")
        try:
            inv_total = float(str(inv_raw).replace(",", "."))
            per = round(inv_total / len(items), 2)
            for i in items:
                i["price"] = per
        except Exception:
            pass

    return {"restaurant": vendor, "datetime": dt, "items": items}

def paraphrase(text: str) -> str:
    if not text:
        return text

    load_qwen_if_needed()

    prompt = f"Paraphrase the following sentence in a clear, friendly way:\n\n{text} .Only generate single sentence of the paraphrased line."
    messages = [{"role": "user", "content": prompt}]


    qwen_input = _qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=True,
    )

    inputs = _qwen_tokenizer(
        [qwen_input],
        return_tensors="pt"
    ).to(_qwen_model.device)

    output_ids = _qwen_model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7
    )[0]
    new_tokens = output_ids[len(inputs.input_ids[0]):].tolist()

    try:
        idx = len(new_tokens) - new_tokens[::-1].index(151668)
    except ValueError:
        idx = 0

    content = _qwen_tokenizer.decode(new_tokens[idx:], skip_special_tokens=True)
    return content.strip()


def parse_receipt(image_bytes: bytes) -> Dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return parse_with_donut(image)


def save_parsed_core(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    session = SessionLocal()
    saved = []
    try:
        restaurant = parsed.get("restaurant")
        dt = parsed.get("datetime", datetime.now())
        for it in parsed.get("items", []):
            item_text = it.get("item", "Unknown")
            price = float(it.get("price") or 0.0)
            emb = embed_text(item_text)
            row = ReceiptItem(item=item_text, price=price, dt=dt, restaurant=restaurant, embedding=emb)
            session.add(row)
            saved.append({"item": item_text, "price": price, "dt": dt.isoformat(), "restaurant": restaurant})
        session.commit()
    finally:
        session.close()
    return saved

# =================================================================
#  Preprocessing
# =================================================================
def interpret_date_from_text(text: str) -> Optional[Tuple[datetime, datetime]]:
    if not text:
        return None

    t = text.lower().strip()
    today = datetime.now()
    today_start = datetime(today.year, today.month, today.day)

    # Simple keywords
    if "today" in t:
        return today_start, today_start + timedelta(days=1)
    if "yesterday" in t:
        d = today - timedelta(days=1)
        start = datetime(d.year, d.month, d.day)
        return start, start + timedelta(days=1)
    if "last" and "days" in t:
        t_temp = t.split()
        for token in t_temp:
            if token.isdigit():
                n_days = int(token)
        d = today - timedelta(days=n_days)
        start = datetime(d.year, d.month, d.day)
        return start, today
    if "this week" in t:
        monday = today_start - timedelta(days=today.weekday())
        return monday, monday + timedelta(days=7)
    if "last week" in t:
        monday = today_start - timedelta(days=today.weekday())
        last_monday = monday - timedelta(days=7)
        return last_monday, last_monday + timedelta(days=7)

    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for i, day_name in enumerate(weekdays):
        if f"last {day_name}" in t:
            monday = today_start - timedelta(days=today.weekday())
            target = monday + timedelta(days=i) - timedelta(days=7)
            start = datetime(target.year, target.month, target.day)
            return start, start + timedelta(days=1)
        if day_name in t:
            monday = today_start - timedelta(days=today.weekday())
            target = monday + timedelta(days=i)
            start = datetime(target.year, target.month, target.day)
            return start, start + timedelta(days=1)


    months = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    t_split = t.split()

    query_day = None
    query_month = None
    query_year = 2025

    for token in t_split:
        if token.isdigit():
            if len(token) == 4:
                query_year = int(tok)
            if len(token) <= 2:
                query_day = int(token)
        if token in months:
            query_month = months[token]

    if query_day and query_month:
        parsed_date = datetime(year=query_year, month=query_month, day=query_day)
        print("PARSED", parsed_date, flush=True)
        return parsed_date, parsed_date + timedelta(days=1)
    
    return None


def get_data_from_range(start: datetime, end: datetime) -> List[ReceiptItem]:
    session = SessionLocal()
    try:
        return (
            session.query(ReceiptItem)
            .filter(ReceiptItem.dt >= start)
            .filter(ReceiptItem.dt < end)
            .all()
        )
    finally:
        session.close()


def parse_image_core(image_bytes: bytes) -> Dict[str, Any]:
    parsed = parse_receipt(image_bytes)
    parsed_for_ui = {
        "restaurant": parsed.get("restaurant"),
        "datetime": (parsed.get("datetime").isoformat() if parsed.get("datetime") else None),
        "items": parsed.get("items", []),
    }
    return {"ui": parsed_for_ui, "raw": parsed}


def save_parsed_from_state_core(parsed_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not parsed_raw:
        return []
    return save_parsed_core(parsed_raw)

# =================================================================
#  Tools
# =================================================================

def what_did_i_buy_core(question: str) -> str:
    date_range = interpret_date_from_text(question)
    if date_range is None:
        return ""  
    start, end = date_range
    rows = get_data_from_range(start, end)
    if not rows:
        return f"No items found for {start.date()}."
    items = sorted({r.item for r in rows})

    if start.date() == end.date():
        return f"You bought: {', '.join(items)} on {start.date()}."
    else:
        return f"You bought: {', '.join(items)} between {start.date()} - {end.date()}."

def where_did_i_buy_core(question: str) -> str:
    if not question or not question.strip():
        return ""

    date_range = interpret_date_from_text(question)
    if date_range is None:
        return ""  # per your rule: if no date indicators, return empty
    start, end = date_range

    rows = get_data_from_range(start, end)
    if not rows:
        return f"No items found for {start.date()}."

    restaurants = sorted({r.restaurants for r in rows})

    if len(restaurants) == 1:
        return f"You bought it at {restaurants[0]}."
    return f"You bought it at: {', '.join(restaurants)}."

def total_spent_core(date_text: str) -> str:
    if not date_text or not date_text.strip():
        return ""

    date_range = interpret_date_from_text(date_text)
    if date_range is None:
        return "" 

    start, end = date_range

    session = SessionLocal()
    try:
        rows = (
            session.query(ReceiptItem)
            .filter(ReceiptItem.dt >= start)
            .filter(ReceiptItem.dt < end)
            .all()
        )
    finally:
        session.close()

    prices = [r.price for r in rows]
    total = sum(prices)

    if start.date() == end.date():
        return f"You spent ${total:.2f} on {start.date()}."
    return f"Your total spending between {start.date()} and {end.date()} is ${total:.2f}."


# =================================================================
#  MCP Server
# =================================================================
mcp = FastMCP("receipt_service")
try:
    mcp.tool(parse_image_core)
    mcp.tool(what_did_i_buy_core)
    mcp.tool(where_did_i_buy_core)
    mcp.tool(total_spent_core)
except Exception:
    pass


def run_mcp_server():
    try:
        mcp.run(host="0.0.0.0", port=8001)
    except Exception:
        pass


# =================================================================
#  UI
# =================================================================
def build_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Receipt Parser")

        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Upload receipt")
                parse_btn = gr.Button("Upload & Parse")
                parse_status = gr.Textbox(label="Parse status", interactive=False)
                parsed_json = gr.JSON(label="Parsed (verify before saving)")
                parsed_state = gr.State(value=None)
                save_btn = gr.Button("Save to DB")
                save_status = gr.Textbox(label="Save status", interactive=False)
            with gr.Column():
                q_input = gr.Textbox(label="Ask (e.g. 'What did I buy last Friday?' or 'How much is Croissant?')")
                ask_btn = gr.Button("Ask")
                answer_out = gr.Textbox(label="Answer", lines= 20,interactive=False)

        def handle_parse(image):
            if image is None:
                return "No image uploaded.", None, None
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            parsed = parse_image_core(bio.getvalue()) 
            ui = parsed["ui"]
            raw = parsed["raw"]
            return "Parsed.", ui, raw

        def handle_save(parsed_raw):
            if not parsed_raw:
                return "Empty Data"
            saved = save_parsed_from_state_core(parsed_raw)
            return f"Saved {len(saved)} items to DB."

        def is_price_query(text: str) -> bool:
            low = (text or "").lower()
            return ("how much" in low) or ("price" in low) or low.strip().startswith("price of") or low.strip().startswith("how much is")

        def is_where_query(text: str) -> bool:
            if not text:
                return False
            low = text.lower()
            return "where" in low and any(w in low for w in ["bought","buy","purchased","purchase","ordered"])

        def is_total_query(text: str) -> bool:
            if not text:
                return False
            low = text.lower()
            return ("total" in low) or ("spent" in low) or ("expense" in low) or (("how much" in low) and any(w in low for w in ["spent","expense","total"]))


        def handle_question(question):
            if not question:
                return paraphrase( "ask something like 'What did I buy last Friday?, 'where did you buy last week', 'how much expense that i have yesterday'")

            if is_where_query(question):
                return paraphrase(where_did_i_buy_core(question))

            if is_total_query(question):
                return paraphrase(total_spent_core(question))

            date_range = interpret_date_from_text(question)
            print(date_range, flush=True)
            if date_range is None:
                return ""  # no date indicators found
            start, end = date_range
            rows = get_data_from_range(start, end)
            if not get_data_from_range:
                if str(start.date()) == str(end.date()):
                    return paraphrase(f"No items found for {start.date()}.")
                else: 
                    return paraphrase(f"No items found for {start.date()} - {end.date()}.")
            items = sorted({r.item for r in rows})
            if str(start.date()) == str(end.date()):
                return paraphrase(f"You bought: {', '.join(items)} on {start.date()}.")
            else:
                return paraphrase(f"You bought: {', '.join(items)} between {start.date()} - {end.date()}.")


        parse_btn.click(handle_parse, inputs=img_in, outputs=[parse_status, parsed_json, parsed_state])
        save_btn.click(handle_save, inputs=parsed_state, outputs=save_status)
        ask_btn.click(handle_question, inputs=q_input, outputs=answer_out)

    return demo


# =================================================================
#  Routes
# =================================================================
fastapi_app = FastAPI(title="Receipt Demo")
gradio_demo = build_gradio_ui()
fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_demo, path="/gradio")


@fastapi_app.get("/")
def root_redirect():
    return RedirectResponse(url="/gradio/")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@fastapi_app.get("/data")
def get_all_data(db: Session = Depends(get_db)):
    rows = db.query(ReceiptItem).all()
    return [
        {
            "id": r.id,
            "item": r.item,
            "price": r.price,
            "datetime": r.dt.isoformat(),
            "restaurant": r.restaurant,
        }
        for r in rows
    ]

@fastapi_app.post("/initial_seed")
def initial_seed(db: Session = Depends(get_db)):

    sample_items = [
        "Matcha Latte", "Chicken Rice Bowl", "Latte"
    ]

    sample_restaurants = [
        "A Restaurant", "S Coffee", "Y Bakery"
    ]

    year = datetime.now().year
    month = 11

    created = []
    for i in range(15):
        item = choice(sample_items)
        price = round(randint(30, 150), 2)
        restaurant = choice(sample_restaurants)

        # Day between 1 → 11
        day = randint(1, 11)
        dt = datetime(year, month, day)

        emb = embed_text(item)

        row = ReceiptItem(
            item=item,
            price=price,
            dt=dt,
            restaurant=restaurant,
            embedding=emb
        )
        db.add(row)
        created.append({
            "item": item,
            "price": price,
            "restaurant": restaurant,
            "datetime": dt.isoformat()
        })

    db.commit()

    return {
        "inserted": len(created),
        "records": created
    }

@fastapi_app.delete("/delete-all")
def delete_all_data(db: Session = Depends(get_db)):
    deleted = db.query(ReceiptItem).delete()
    db.commit()
    return {"deleted": int(deleted)}

@fastapi_app.on_event("startup")
def on_startup():
    init_db()
    try:
        load_donut_if_needed()
        load_qwen_if_needed()
        get_embedding_model()
        print("Models loaded.")
    except Exception as e:
        print("Model loading warning:", e)

    threading.Thread(target=run_mcp_server, daemon=True).start()

app = fastapi_app


if __name__ == "__main__":
    init_db()
    try:
        load_donut_if_needed()
        get_embedding_model()
    except Exception:
        pass
    threading.Thread(target=run_mcp_server, daemon=True).start()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
