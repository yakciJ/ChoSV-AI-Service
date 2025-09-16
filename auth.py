import os
from fastapi import Request, HTTPException

API_KEY = os.getenv("API_KEY")  # nên lấy từ biến môi trường thay vì hardcode

async def api_key_middleware(request: Request, call_next):
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    response = await call_next(request)
    return response