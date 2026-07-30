import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.approvals.store import ApprovalNotFoundError

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Mini Agentic ERP Assistant")

# Dev-only: allows the Next.js dev server (different origin/port) to call
# this API directly, including reading the /chat/stream SSE response.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(ApprovalNotFoundError)
def approval_not_found_handler(request: Request, exc: ApprovalNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})
