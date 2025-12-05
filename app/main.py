from fastapi import FastAPI 
from app.routers import tokens

app = FastAPI()

app.include_router(tokens.router)


@app.get("/")
def read_root():
    return {"msg": "Gen AI Lab is UP!!!"}