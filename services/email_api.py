"""Admin email API service entrypoint."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_routes.email import router as email_router

from services.api_common import parse_cors_origins, setup_service_logging

setup_service_logging()

app = FastAPI(title="Cyber-Lenin Email API")
app.include_router(email_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins("EMAIL_CORS_ORIGINS"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "service": "leninbot-email-api"}


if __name__ == "__main__":
    print("Email API server starting on 127.0.0.1:8002")
    uvicorn.run(app, host="127.0.0.1", port=8002)
