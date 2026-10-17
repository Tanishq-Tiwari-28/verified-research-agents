# Verified Research Agents

Chrome side-panel extension + Python backend. Three Gemini agents (Retriever, Synthesizer,
Fact-Checker) answer questions from live web search, and every answer is cross-checked
against its sources before it reaches you.

## Run it (about 3 minutes)

1. Backend
   ```bash
   cd backend
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env                                  # paste your GEMINI_API_KEY (aistudio.google.com/apikey)
   uvicorn server:app --port 8000
   ```
2. Extension: open `chrome://extensions`, turn on Developer mode, click **Load unpacked**, pick the `extension/` folder.
   Click the extension icon to open the side panel.
3. Eval (run from the project root, with GEMINI_API_KEY set or backend/.env present)
   ```bash
   python eval/run_eval.py --mode baseline
   python eval/run_eval.py --mode full
   python eval/run_eval.py --compare eval/results_baseline.json eval/results_full.json
   ```

## Architecture

question -> Retriever (web_search / fetch_page loop) -> Evidence store (S1..Sn)
         -> Synthesizer (cited draft, evidence only)
         -> Fact-Checker (claim-by-claim check, may search again) -> pass | revise (max 2 rounds)

## The recency fix

Failure: "most recent X" answered with an older X even though search ran. The model trusted its memory
and never saw today's date. Fixes (toggle with `Config.date_aware`): inject today's date, require a news
search with the current year in the query, extract publication dates from pages, tell the Synthesizer to
cite the date it relies on, and have the Fact-Checker flag "outdated" claims and re-search them.
