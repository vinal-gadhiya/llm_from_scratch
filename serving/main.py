from fastapi import FastAPI

from serving import inference

app = FastAPI(title="Chat App")

app.include_router(inference.router)