from fastapi import FastAPI 
from app.routers import tokens,tokeniser

app = FastAPI()

app.include_router(tokens.router)
app.include_router(tokeniser.router)


@app.get("/")
def read_root():
    return {"msg": "Gen AI Lab is UP!!!"}