from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import duckdb
from sentence_transformers import SentenceTransformer


DB_PATH = "vectordb.duckdb"
MODEL_NAME = "all-MiniLM-L6-v2"


class RequestSchema(BaseModel):
    text: str

class ResponseSchema(BaseModel):
    text: str
    similarity: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()

    # safe norms
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a @ b) / (na * nb))

class VectorDB:
    def __init__(self, db_path: str = DB_PATH, model_name: str = MODEL_NAME):
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(database=str(self.db_path), read_only=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_embedding = SentenceTransformer(model_name, device=device)
        self.initialize_table()

    def initialize_table(self):
        """Create table if not exists and seed initial data if empty."""
        create_table = """
        CREATE TABLE IF NOT EXISTS vectors (
            text VARCHAR,
            embedding FLOAT[]
        );
        """
        self.conn.execute(create_table)

        #check if table is empty/not

        count = self.conn.execute("SELECT COUNT(*) FROM vectors;").fetchone()
        if count[0] == 0:
            self.seed_initial_data()
        
    def seed_initial_data(self):
        fruits = ["apple","banana","orange","pineaple","mango"]
        embeddings = self.model_embedding.encode(fruits, convert_to_numpy=True)

        for fruit, embedding in zip(fruits,embeddings):
            emb = embedding.astype(float).tolist()

            self.conn.execute("INSERT INTO vectors (text, embedding) VALUES (?, ?)", (fruit, emb))
    
    def search(self, query: str):

        if not query:
            raise ValueError("Empty Query")
        
        query_emb = self.model_embedding.encode([query], convert_to_numpy=True)[0]
        all_data = self.conn.execute("SELECT text, embedding FROM vectors").fetchall()


        highest_similarity = -1.0
        most_similar_text = ""

        for text, emb in all_data:
            emb_vec = np.array(emb, dtype=np.float32)
            similarity = cosine_similarity(query_emb, emb_vec)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_text = text
        
        return most_similar_text, highest_similarity


app = FastAPI(title="simple vectorDB")
db = VectorDB()

@app.post("/search", response_model=ResponseSchema)
def search(req: RequestSchema):
    try:
        text_result, sim_result = db.search(req.text)
        return ResponseSchema(text=text_result, similarity=sim_result)
    except:
        raise HTTPException(status_code=500)

                





