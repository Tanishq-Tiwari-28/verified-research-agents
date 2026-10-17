import json
import queue
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import Config, answer_question

app = FastAPI(title="Verified Research Agents")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(body: Ask):
    return answer_question(body.question, Config())


@app.post("/ask_stream")
def ask_stream(body: Ask):
    """NDJSON stream: one line per agent event, then a final line with the full result."""
    q = queue.Queue()

    def worker():
        try:
            res = answer_question(body.question, Config(), on_event=lambda e: q.put({"event": "trace", **e}))
            q.put({"event": "final", "result": res})
        except Exception as e:
            q.put({"event": "error", "message": f"{type(e).__name__}: {e}"})
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while (item := q.get()) is not None:
            yield json.dumps(item) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
