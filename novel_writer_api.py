
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_routes.writer import router as writer_router

from api_common import parse_cors_origins, setup_service_logging

setup_service_logging()

app = FastAPI(title="Cyber-Lenin Novel Writer API")
app.include_router(writer_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins("WRITER_CORS_ORIGINS"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "novel-writer-api"}


if __name__ == "__main__":
    print("Novel writer API server starting on 127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
