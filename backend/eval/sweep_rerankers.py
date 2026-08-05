"""
Reranker sweep — compares several cross-encoder rerankers on the SAME questions.

For each question we fetch one candidate pool (fetch_k) with the vector search,
then let every reranker reorder that same pool (fair, apples-to-apples). The
"baseline" arm keeps the vector order. Each arm's top-k is generated + judged.

Reuses the cached questions and helpers from evaluate_rag.

Run from repo root:
    ./backend/.venv/bin/python -m backend.eval.sweep_rerankers
"""
import json
import os
import time

from sentence_transformers import CrossEncoder

from backend.app.generator import generate_answer
from backend.eval.evaluate_rag import (
    QUESTIONS_FILE, K, FETCH_K,
    build_store, gold_rank, reciprocal_rank, judge,
)

SWEEP_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "sweep_results.json")

# label -> HF model id (None = baseline, keep vector-search order)
MODELS = {
    "baseline (no rerank)": None,
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
}


def rerank(model, query, candidates):
    """Reorder candidates by cross-encoder score; return (top-k docs, ms spent)."""
    t0 = time.perf_counter()
    scores = model.predict([(query, d.page_content) for d in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
    ms = (time.perf_counter() - t0) * 1000.0
    return [d for d, _ in ranked[:K]], ms


def evaluate_arm(label, model, questions, pools):
    """Run one reranker over every question's candidate pool; return metrics."""
    n = len(questions)
    hit = mrr = corr = faith = lat = 0.0
    judged = 0

    for q, pool in zip(questions, pools):
        if model is None:
            docs, ms = pool[:K], 0.0          # baseline = vector order
        else:
            docs, ms = rerank(model, q["question"], pool)

        rank = gold_rank(docs, q["gold_source"], q["gold_page"])
        hit += 1 if rank else 0
        mrr += reciprocal_rank(rank)
        lat += ms

        answer = generate_answer(q["question"], docs)["answer"]
        context = "\n".join(d.page_content for d in docs)
        verdict = judge(q["question"], q["reference_answer"], context, answer)
        if verdict:
            corr += verdict["correctness"]
            faith += verdict["faithfulness"]
            judged += 1

    return {
        "hit@k": hit / n,
        "mrr": mrr / n,
        "correctness": corr / judged if judged else 0.0,
        "faithfulness": faith / judged if judged else 0.0,
        "rerank_ms": lat / n,
    }


def print_table(results):
    cols = ["hit@k", "mrr", "correctness", "faithfulness", "rerank_ms"]
    head = ["Hit@%d" % K, "MRR", "Correct", "Faithful", "Rerank ms"]
    print("\n" + "=" * 92)
    print(f"  {'model':<26}" + "".join(f"{h:>13}" for h in head))
    print("-" * 92)
    for label, m in results.items():
        cells = [
            f"{m['hit@k']:.3f}", f"{m['mrr']:.3f}", f"{m['correctness']:.2f}",
            f"{m['faithfulness']:.2f}", f"{m['rerank_ms']:.1f}",
        ]
        print(f"  {label:<26}" + "".join(f"{c:>13}" for c in cells))
    print("=" * 92)


def main():
    print("Building vector store from books ...")
    store, _ = build_store()

    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Fetch each question's candidate pool ONCE; every reranker reorders it.
    pools = [store.similarity_search(q["question"], k=FETCH_K) for q in questions]

    results = {}
    for label, model_id in MODELS.items():
        print(f"\n>>> {label}")
        model = CrossEncoder(model_id) if model_id else None
        results[label] = evaluate_arm(label, model, questions, pools)

    print_table(results)
    with open(SWEEP_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results -> {SWEEP_RESULTS_FILE}")


if __name__ == "__main__":
    main()
