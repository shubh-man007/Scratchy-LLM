from fastapi import FastAPI
from typing import Dict

app = FastAPI(
    title="Musings",
    description="Prep rev"
)

@app.get("/")
def get_health() -> Dict:
    return {"health": "OK"}

# uvicorn app:app --reload --host 0.0.0.0 --port 2020
