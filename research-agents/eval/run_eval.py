"""30-question eval harness.

  python eval/run_eval.py --mode baseline      # single pass: no date-awareness, no fact-checker
  python eval/run_eval.py --mode full          # all fixes on
  python eval/run_eval.py --compare eval/results_baseline.json eval/results_full.json

Questions with "expected": null are time-sensitive. Look the answer up TODAY, fill it in,
and re-run both modes so the comparison is on the same labelled set.
An LLM judge grades each answer against the gold label and tags a failure type."""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from google.genai import types  # noqa: E402

from agents import Config, _extract_json, _generate, _text, answer_question  # noqa: E402

HERE = os.path.dirname(__file__)
MODES = {
    "baseline": Config(date_aware=False, fact_check=False),
    "nocheck": Config(date_aware=True, fact_check=False),   # ablation: date fixes only
    "full": Config(date_aware=True, fact_check=True),
}
JUDGE = """You grade a QA system. Question: {q}
Gold answer: {gold}
System answer: {ans}

Is the system answer correct, i.e. consistent with the gold answer and not contradicting it?
Reply ONLY with JSON: {"correct": true|false, "failure_type": "none"|"stale_answer"|"wrong_entity"|"hedged_no_answer"|"unsupported_claim"|"other", "reason": "<=15 words"}
Use "stale_answer" when the system gave an older instance of a 'most recent/current' fact."""


def judge(q, gold, ans):
    prompt = JUDGE.replace("{q}", q).replace("{gold}", gold).replace("{ans}", ans or "(empty)")
    out = _extract_json(_text(_generate(prompt, types.GenerateContentConfig(temperature=0))))
    return out or {"correct": False, "failure_type": "other", "reason": "judge parse error"}


def run_one(item, cfg):
    t = time.time()
    try:
        r = answer_question(item["q"], cfg)
        j = judge(item["q"], item["expected"], r["answer"])
        return {**item, "answer": r["answer"], "verified": r["verified"], "revisions": r["revisions"],
                "correct": bool(j.get("correct")), "failure_type": j.get("failure_type"),
                "reason": j.get("reason"), "secs": round(time.time() - t, 1)}
    except Exception as e:
        return {**item, "answer": "", "correct": False, "failure_type": "error", "reason": str(e)[:120], "secs": 0}


def compare(a_path, b_path):
    a = {r["id"]: r for r in json.load(open(a_path))["results"]}
    b = {r["id"]: r for r in json.load(open(b_path))["results"]}
    for i in sorted(a.keys() & b.keys()):
        if a[i]["correct"] != b[i]["correct"]:
            arrow = "FIXED " if b[i]["correct"] else "BROKE "
            print(f"{arrow} #{i} {a[i]['q']}\n   A: {a[i]['answer'][:110]!r}\n   B: {b[i]['answer'][:110]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="full")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)

    items = json.load(open(os.path.join(HERE, "questions.json")))
    labelled = [i for i in items if i["expected"]]
    skipped = len(items) - len(labelled)
    if skipped:
        print(f"WARNING: {skipped} question(s) have no gold label and are skipped. Fill in 'expected' to include them.\n")

    cfg = MODES[args.mode]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda it: run_one(it, cfg), labelled))

    ok = sum(r["correct"] for r in results)
    print(f"mode={args.mode}  accuracy: {ok}/{len(results)} = {ok / len(results):.0%}")
    fails = [r for r in results if not r["correct"]]
    print("failure types:", dict(collections.Counter(r["failure_type"] for r in fails)))
    for r in fails:
        print(f"  #{r['id']} [{r['failure_type']}] {r['q']}\n     got: {r['answer'][:110]!r}\n     why: {r['reason']}")

    out = os.path.join(HERE, f"results_{args.mode}.json")
    json.dump({"mode": args.mode, "accuracy": ok / len(results), "n": len(results), "results": results},
              open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
