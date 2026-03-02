#!/usr/bin/env python3
"""ContentAI Pro — Launch script."""
import os
import uvicorn

if __name__ == "__main__":
    debug = os.getenv("DEBUG", "true").lower() == "true"
    uvicorn.run(
        "contentai_pro.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=debug,
        log_level="debug" if debug else "info",
    )
