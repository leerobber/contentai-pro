#!/usr/bin/env python3
"""ContentAI Pro — Launch script."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "contentai_pro.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
