from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.model import router as Model


# Create FastAPI app
app = FastAPI(
    title = "FireRedASR API",
    description = "FireRedASR automatic speech recognition microservice",
    version = "1.0.0",
)

# Add Middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(Model)


@app.get("/", status_code=200, tags=["Root"])
async def root():
    """
    Return message from container to check if it is running.
    """
    return {"detail": "FireRedASR API is running"} 