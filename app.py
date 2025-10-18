import os
import uvicorn
import logging
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from productController import router as insert_router
from search import router as search_router
from dotenv import load_dotenv


# ===== Logging setup =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ===== Load API_KEY =====
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY is not set in the environment variables!")
else:
    logger.info("API_KEY loaded successfully.")

# ===== Init app =====
app = FastAPI()

# ===== Middleware =====
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    key = request.headers.get("x-api-key")
    print(key)
    print(API_KEY)
    if not key:
        return JSONResponse(status_code=403, content={"detail": "API key missing"})
    if key != API_KEY:
        return JSONResponse(status_code=403, content={"detail": "Unauthorized"})
    
    logger.info(f"Valid API key received: {key}")
    response = await call_next(request)
    return response

# ===== Routers =====
app.include_router(insert_router)
app.include_router(search_router)

# ===== Run =====
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
