from fastapi import FastAPI, APIRouter
import uvicorn
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from router.item.item import router as item_router
from router.models.models import router as models_router
from router.files.files import router as files_router
from router.limiter import limiter

import os
from dotenv import load_dotenv

load_dotenv()


test_router = APIRouter()
app = FastAPI()  

@test_router.get("/posts")
async def posts() -> dict:
    return {"posts": "test"}


app.include_router(item_router)
app.include_router(models_router)
app.include_router(files_router)
app.include_router(test_router)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/")
def read_root() -> str:
    return "Server is running."


if __name__ == "__main__":
    port = os.getenv("PORT")
    host = os.getenv("HOST")
    uvicorn.run(
        app, 
        host = host, 
        port = port, 
        log_level = "info"
    )
