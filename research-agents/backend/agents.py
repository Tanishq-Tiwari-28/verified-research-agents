"""Three cooperating agents on top of the Gemini function-calling loop:
  Retriever    -> searches / reads the live web (tool-calling loop)
  Synthesizer  -> writes a cited draft using only the gathered evidence
  Fact-Checker -> decomposes the draft into claims, verifies each against the
                  evidence (and re-searches on its own when needed), then
                  passes or sends the draft back for revision."""
import datetime
import json
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import Evidence, Toolbox

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
_client = None


def client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


@dataclass
class Config:
    date_aware: bool = True      # inject today's date + recency rules (the fix for stale "most recent" answers)
    fact_check: bool = True      # run the Fact-Checker agent
    max_search_steps: int = 6
    max_revisions: int = 2


class Trace:
    """Collects (and optionally streams) every agent action for the UI."""

    def __init__(self, cb=None):
        self.events, self.cb, self.t0 = [], cb, time.time()

    def emit(self, agent, kind, detail):
        e = {"agent": agent, "kind": kind, "detail": detail, "t": round(time.time() - self.t0, 1)}
        self.events.append(e)
        if self.cb:
            self.cb(e)


def _today():
    return datetime.date.today().strftime("%A, %B %d, %Y")


def _fill(s, **kw):
    for k, v in kw.items():
        s = s.replace("{" + k + "}", v)
    return s


# ---------------------------------------------------------------- prompts
DATE_BLOCK = """Today's date is {today}. Your internal knowledge is out of date, so treat it as unreliable for anything that can change.
For questions with words like latest, most recent, current, newest, now, or any winner/holder/record-type fact:
use kind='news' at least once, put the relevant year ({year}) in the query, check every source's published date,
and keep searching until you have a source from the last few months or have confirmed nothing newer exists. Never answer from memory."""

RETRIEVER_SYS = """You are the Retriever agent in a research pipeline. You do NOT answer the question; you gather evidence.
{date_block}
Process: (1) run 1-3 focused web_search queries (short and specific); (2) fetch_page on the 2-3 most relevant/authoritative
results to get full text and publication dates; (3) if results are thin or conflicting, reformulate and search again.
When you have enough evidence, stop calling tools and reply with one line: DONE: <what you found>."""

SYNTH_SYS = """You are the Synthesizer agent. Answer the question using ONLY the numbered evidence provided.
Rules: cite every factual claim with source ids like [S1] or [S1][S3]; use no outside knowledge; if sources conflict, prefer
the most recent authoritative one and say so briefly; if the evidence does not answer the question, say exactly what is missing;
be concise (2-5 sentences).{date_line}"""

CHECK_SYS = """You are the Fact-Checker agent. {date_block}
You get a question, a draft answer with [S#] citations, and the evidence it was built from.
1. Break the draft into atomic factual claims.
2. For each claim, check that the cited evidence text actually states it (plausible is not enough).
3. If a claim is unsupported, contradicted, or time-sensitive ('latest', 'current', 'most recent') and may be outdated,
   run a targeted web_search/fetch_page to settle it before deciding. New sources get new S ids you can cite.
4. Verdict is "pass" only if EVERY claim is supported and none is outdated.
Final output must be ONLY this JSON, no prose, no code fences:
{"verdict":"pass"|"fail","claims":[{"claim":"...","status":"supported"|"unsupported"|"contradicted"|"outdated","source_ids":["S1"],"note":"..."}],"summary":"one sentence on what must change"}"""


# ---------------------------------------------------------------- Gemini helpers
def _generate(contents, config):
    for i in range(5):
        try:
            return client().models.generate_content(model=MODEL, contents=contents, config=config)
        except Exception as e:
            msg = str(e)
            retryable = any(k in msg for k in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if i == 4 or not retryable:
                raise
            time.sleep(2 ** i)


def _text(resp):
    if not resp.candidates or not resp.candidates[0].content or not resp.candidates[0].content.parts:
        return ""
    return "".join(p.text for p in resp.candidates[0].content.parts
                   if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()


def run_agent(name, system, user, toolbox, trace, max_steps):
    """The function-calling loop: model asks for a tool -> we run it -> feed the result back -> repeat."""
    tools = [types.Tool(function_declarations=toolbox.declarations)]
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=0.2, tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    contents = [types.Content(role="user", parts=[types.Part(text=user)])]

    for _ in range(max_steps):
        resp = _generate(contents, cfg)
        if not resp.candidates or not resp.candidates[0].content or not resp.candidates[0].content.parts:
            break
        content = resp.candidates[0].content
        contents.append(content)  # keep model turn intact (preserves thought signatures)
        calls = [p.function_call for p in content.parts if p.function_call]
        if not calls:
            return _text(resp)
        results = []
        for call in calls:
            args = dict(call.args or {})
            trace.emit(name, "tool", f"{call.name}({json.dumps(args)[:140]})")
            results.append(types.Part.from_function_response(
                name=call.name, response={"result": toolbox.call(call.name, args)}))
        contents.append(types.Content(role="user", parts=results))

    # tool budget exhausted: force a final answer with tool calling switched off
    contents[-1].parts.append(types.Part(text="Tool budget used up. Give your final output now; do not call tools."))
    final_cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=0.2, tools=tools,
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")))
    return _text(_generate(contents, final_cfg))


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- agents
def _date_block(cfg):
    if not cfg.date_aware:
        return ""
    return _fill(DATE_BLOCK, today=_today(), year=str(datetime.date.today().year))


def retrieve(question, ev, cfg, trace):
    trace.emit("Retriever", "stage", "Searching the live web")
    system = _fill(RETRIEVER_SYS, date_block=_date_block(cfg))
    notes = run_agent("Retriever", system, question, Toolbox(ev), trace, cfg.max_search_steps)
    trace.emit("Retriever", "info", f"{len(ev.items)} sources collected. {notes[:120]}")


def synthesize(question, ev, cfg, trace, prior=None, report=None):
    trace.emit("Synthesizer", "stage", "Writing cited draft" if not prior else "Revising draft")
    date_line = (f" Today's date is {_today()}; for 'latest/current' questions, state the date of the source you rely on."
                 if cfg.date_aware else "")
    prompt = f"Question: {question}\n\nEvidence:\n{ev.render()}"
    if prior:
        prompt += (f"\n\nYour previous draft:\n{prior}\n\nFact-checker report:\n{json.dumps(report)}\n\n"
                   "Rewrite the answer fixing every flagged claim. Remove or soften anything unsupported.")
    resp = _generate(prompt, types.GenerateContentConfig(
        system_instruction=_fill(SYNTH_SYS, date_line=date_line), temperature=0.2))
    return _text(resp)


def fact_check(question, draft, ev, cfg, trace):
    trace.emit("Fact-Checker", "stage", "Verifying claims against sources")
    system = _fill(CHECK_SYS, date_block=_date_block(cfg))
    user = f"Question: {question}\n\nDraft answer:\n{draft}\n\nEvidence:\n{ev.render()}"
    raw = run_agent("Fact-Checker", system, user, Toolbox(ev), trace, 4)
    report = _extract_json(raw) or {"verdict": "fail", "claims": [], "summary": "Checker output was not parseable."}
    bad = [c for c in report.get("claims", []) if c.get("status") != "supported"]
    trace.emit("Fact-Checker", "result", f"verdict={report.get('verdict')} ({len(bad)} flagged claim(s))")
    return report


def answer_question(question, cfg=None, on_event=None):
    cfg = cfg or Config()
    trace, ev = Trace(on_event), Evidence()
    retrieve(question, ev, cfg, trace)
    draft = synthesize(question, ev, cfg, trace)

    report, revisions = None, 0
    if cfg.fact_check:
        for attempt in range(cfg.max_revisions + 1):
            report = fact_check(question, draft, ev, cfg, trace)
            if report.get("verdict") == "pass" or attempt == cfg.max_revisions:
                break
            draft = synthesize(question, ev, cfg, trace, prior=draft, report=report)
            revisions += 1

    verified = bool(report and report.get("verdict") == "pass")
    return {"question": question, "answer": draft, "verified": verified if cfg.fact_check else None,
            "revisions": revisions, "fact_check": report, "sources": ev.cited(draft),
            "trace": trace.events}
