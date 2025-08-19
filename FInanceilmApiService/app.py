from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
from pydantic import BaseModel
import os
import json
import asyncio
import warnings

from dotenv import load_dotenv
load_dotenv()

# --- your existing imports ---
from src.models import QuestionInput
from src.services.openaiservice import ChatCompletionChunk
from src.financeilm import FinanceILM
from src.services.v1 import completion_v1, completion_v1_stream
from src.services.logservice import logging

warnings.filterwarnings("ignore")

# =========================
# Security: Bearer Token
# =========================
security = HTTPBearer(auto_error=False)  # let us raise our own 401

API_TOKEN = os.getenv("FINANCEILM_API_TOKEN")
if not API_TOKEN:
    logging.warning("FINANCEILM_API_TOKEN is not set. All requests will fail with 401.")

def require_bearer_token(
    creds: HTTPAuthorizationCredentials = Depends(security)
) -> None:
    """
    Dependency to require Authorization: Bearer <token>.
    Validates against FINANCEILM_API_TOKEN.
    """
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized: Bearer token required")
    token = creds.credentials
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")


# =========================
# App & CORS
# =========================
app = FastAPI(title="FINANCEILM", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],  # ensure Authorization header is allowed
)

# =========================
# Models (legacy support)
# =========================
class LegacyQuestionInput(BaseModel):
    Question: str
    queries: List[dict] = []
    flag: Optional[str] = "False"
    source: Optional[str] = "site"

chatIlm = FinanceILM()

# =========================
# Routes
# =========================

@app.post("/api/v1/context-in-usage", dependencies=[Depends(require_bearer_token)])
async def completionWithContextInUsage(data: QuestionInput):
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=400,
            detail="FinanceILM: Please provide a message to generate a completion",
        )

    try:
        context = await chatIlm.get_context(data.messages[-1].content, data.referrer)
        with open("context.txt", "w", encoding="utf-8") as file:
            file.write(context[0])
    except Exception as e:
        logging.error(f"Error retrieving context from Chromadb: {e}")
        raise HTTPException(status_code=500, detail="Internal server error: Error retrieving context")

    def parse_stream(stream):
        for chunk in stream:
            # normalize OpenAI chunk to JSON
            chunk_json_obj = json.loads(ChatCompletionChunk(**chunk.__dict__).model_dump_json())
            # include context when usage arrives (tail of stream)
            if chunk_json_obj.get("usage"):
                chunk_json_obj["context"] = {"text": context[0], "link": context[1]}
            yield f"{json.dumps(chunk_json_obj)}\n\n"

    if getattr(data, "stream", False):
        stream = completion_v1_stream(context, data.messages, data.referrer)
        return StreamingResponse(parse_stream(stream), media_type="application/json")
    else:
        res = completion_v1(context, data.messages, data.referrer)
        res = dict(res)
        res.update({"context": {"text": context[0], "link": context[1]}})
        return res


# Optional: a public healthcheck if you want something unprotected
# @app.get("/healthz")
# async def healthz():
#     return {"status": "ok"}
# --- add this public healthcheck (no auth) ---
@app.get("/healthz")
async def healthz():
    return {"ok": True}

# (optional) add a friendly public landing page
@app.get("/public")
async def public_root():
    return {"status": "ok", "service": "FinanceILM API Version 1.0.0"}

# keep this protected root (requires Authorization: Bearer <token>)
@app.get("/", dependencies=[Depends(require_bearer_token)])
async def home():
    return {"status": "ok", "service": "FinanceILM API"}

if __name__ == "__main__":
    import uvicorn
    # Use --proxy-headers and --forwarded-allow-ips if you're behind a proxy
    uvicorn.run(app, host="0.0.0.0", port=8000)
