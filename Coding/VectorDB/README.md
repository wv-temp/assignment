## Installation
pip install -r requirements.txt

If using GPU, install torch separately (example for CUDA 12.x):
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu122

## Run Server
uvicorn main:app --reload

## API Documentation
Open in browser:
http://localhost:8000/docs