# Receipt Parser

## Installation
```
pip install -r requirements.txt
```


If using GPU:
```
pip install torch --index-url https://download.pytorch.org/whl/cu122
```


## Run (Local)
```
uvicorn app:app --reload
```


## Run with Docker
```
docker-compose up --build
```

## UI

Gradio interface:
```
http://localhost:8000/gradio
```

## API Endpoints
```
GET /
```
Redirects to the Gradio UI.

```
GET /data
```
Returns all stored receipt items.

```
POST /initial_seed
```
Seeds the database with sample items.

```
DELETE /delete-all
```
Removes all items from the database.


## MCP Tools (internal)
- parse_image_core  
- what_did_i_buy_core  
- where_did_i_buy_core  
- total_spent_core  


## Future Goals
- If it's available (and possible), incorporate a more state-of-the-art LLM that has more capabilities
