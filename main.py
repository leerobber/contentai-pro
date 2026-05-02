from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from db.models import init_db
from config.settings import DEBUG

app = FastAPI(
    title="ContentAI Pro",
    description="AI-powered content generation API",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()

app.include_router(router)

@app.get("/")
async def root():
    return {"service": "ContentAI Pro", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=DEBUG)
