"""
ContentAI Pro — Main Entry Point
SovereignNation | Built by GH05T3
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from config.settings import settings

app = FastAPI(
    title="ContentAI Pro",
    description="AI-powered content creation platform by SovereignNation",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"status": "online", "product": "ContentAI Pro", "company": "SovereignNation"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
