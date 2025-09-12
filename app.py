from fastapi import FastAPI
import uvicorn
from insert import router as insert_router
from search import router as search_router

app = FastAPI()
app.include_router(insert_router)
app.include_router(search_router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)