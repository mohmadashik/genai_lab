from fastapi import FastAPI 
from app.routers import tokens,tokeniser,embeddings,vector_database

app = FastAPI()

app.include_router(tokens.router)
app.include_router(tokeniser.router)
app.include_router(embeddings.router)
app.include_router(vector_database.router)


@app.get("/")
def read_root():
    return {"msg": "Gen AI Lab is UP!!!"}
