"""
RAG evaluation harness — measures the cross-encoder reranker's impact.

Runs the REAL pipeline (load_pdf -> chunk -> FAISS -> retrieve -> generate)
twice per question: reranker OFF (baseline) and ON. Reports, per arm:
  Hit@k, MRR, answer correctness (1-5), faithfulness (1-5), retrieval latency.

Questions are auto-generated from the test PDF with an LLM (gpt-4o-mini) and
cached to questions.json — each question records the source chunk's page as the
gold page, giving us a retrieval ground-truth label for free.

Run from repo root:
    ./backend/.venv/bin/python -m backend.eval.evaluate_rag
"""
import json
import os
import random
import time

from dotenv import load_dotenv

# Load backend/.env so OPENAI_API_KEY is set (same as backend/app/main.py does).
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from backend.app.pdf_loader import load_pdf
from backend.app.chunker import chunk_text
from backend.app.rag import create_vector_store, retrieve_chunks
from backend.app.generator import generate_answer, _get_client

# --- config (all easy to change) ---------------------------------------------
HERE = os.path.dirname(__file__)
PDF_PATH = os.path.join(
    HERE, "..", "uploads", "Grokking_Machine_Learning_-_Luis_Serrano.pdf"
)
QUESTIONS_FILE = os.path.join(HERE, "questions.json")
RESULTS_FILE = os.path.join(HERE, "results.json")

N_QUESTIONS = 18
K = 6           # chunks fed to the LLM
FETCH_K = 20    # candidate pool the reranker re-scores
JUDGE_MODEL = "gpt-4o-mini"
SEED = 42


# --- retrieval metrics (pure) ------------------------------------------------
def gold_rank(retrieved_docs, gold_page):
    """1-based rank of the gold page in the retrieved list, or None if absent."""
    for i, doc in enumerate(retrieved_docs):
        if doc.metadata.get("page") == gold_page:
            return i + 1
    return None


def reciprocal_rank(rank):
    return 1.0 / rank if rank else 0.0


# --- question generation (cached) --------------------------------------------
def generate_questions(chunks):
    """Sample substantial chunks and have the LLM write a Q + reference answer."""
    client = _get_client()
    substantial = [c for c in chunks if len(c["content"]) > 300]
    random.seed(SEED)
    sample = random.sample(substantial, min(N_QUESTIONS, len(substantial)))

    questions = []
    for c in sample:
        prompt = (
            "From the textbook passage below, write ONE specific factual "
            "question that can be answered using only this passage, plus a "
            "concise reference answer. Avoid vague questions. Respond as JSON: "
            '{"question": "...", "answer": "..."}\n\nPassage:\n' + c["content"]
        )
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  question-gen skipped for page {c['page']}: {e}")
            continue

        if data.get("question") and data.get("answer"):
            questions.append({
                "question": data["question"].strip(),
                "reference_answer": data["answer"].strip(),
                "gold_page": c["page"],
            })

    with open(QUESTIONS_FILE, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Generated {len(questions)} questions -> {QUESTIONS_FILE}")
    return questions


def load_or_generate_questions(chunks):
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE) as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} cached questions from {QUESTIONS_FILE}")
        return questions
    print("No cached questions — generating from the PDF...")
    return generate_questions(chunks)


# --- LLM-as-judge ------------------------------------------------------------
def judge(question, reference_answer, context, answer):
    """
    One judge call scoring both axes 1-5:
      correctness  — answer agrees with the reference answer
      faithfulness — answer is supported by the retrieved context (no hallucination)
    Returns {"correctness": int, "faithfulness": int} or None on failure.
    """
    prompt = (
        "You are grading a RAG system's answer. Score two things on a 1-5 "
        "integer scale (5 best):\n"
        "- correctness: does the ANSWER match the REFERENCE ANSWER?\n"
        "- faithfulness: is the ANSWER supported by the CONTEXT, with no "
        "invented facts?\n"
        'Respond as JSON: {"correctness": N, "faithfulness": N}\n\n'
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}"
    )
    try:
        resp = _get_client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "correctness": int(data["correctness"]),
            "faithfulness": int(data["faithfulness"]),
        }
    except Exception as e:
        print(f"  judge failed: {e}")
        return None


# --- one arm of the A/B (baseline or reranked) -------------------------------
def run_arm(question, store, gold_page, rerank):
    t0 = time.perf_counter()
    docs = retrieve_chunks(question, store, k=K, fetch_k=FETCH_K, rerank=rerank)
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    rank = gold_rank(docs, gold_page)
    context = "\n".join(d.page_content for d in docs)
    answer = generate_answer(question, docs)["answer"]
    return {
        "docs": docs,
        "context": context,
        "answer": answer,
        "rank": rank,
        "hit": rank is not None,
        "rr": reciprocal_rank(rank),
        "retrieval_ms": retrieval_ms,
    }


def summarize(rows, arm):
    n = len(rows)
    hit = sum(r[arm]["hit"] for r in rows) / n
    mrr = sum(r[arm]["rr"] for r in rows) / n
    lat = sum(r[arm]["retrieval_ms"] for r in rows) / n
    judged = [r[arm]["judge"] for r in rows if r[arm]["judge"]]
    corr = sum(j["correctness"] for j in judged) / len(judged) if judged else 0.0
    faith = sum(j["faithfulness"] for j in judged) / len(judged) if judged else 0.0
    return {
        "hit@k": hit, "mrr": mrr, "correctness": corr,
        "faithfulness": faith, "retrieval_ms": lat,
    }


def print_report(base, rerank):
    def row(label, b, r, fmt):
        d = r - b
        sign = "+" if d >= 0 else ""
        print(f"  {label:<16} {fmt.format(b):>10} {fmt.format(r):>10} {sign}{fmt.format(d):>10}")

    print("\n" + "=" * 52)
    print(f"  {'metric':<16} {'baseline':>10} {'reranked':>10} {'delta':>11}")
    print("-" * 52)
    row("Hit@%d" % K, base["hit@k"], rerank["hit@k"], "{:.3f}")
    row("MRR", base["mrr"], rerank["mrr"], "{:.3f}")
    row("Correctness", base["correctness"], rerank["correctness"], "{:.2f}")
    row("Faithfulness", base["faithfulness"], rerank["faithfulness"], "{:.2f}")
    row("Retrieval ms", base["retrieval_ms"], rerank["retrieval_ms"], "{:.1f}")
    print("=" * 52)


def _selftest():
    """Guard the pure metric logic — the piece most likely to silently break."""
    class D:
        def __init__(self, page):
            self.metadata = {"page": page}
    docs = [D(5), D(2), D(9)]
    assert gold_rank(docs, 2) == 2
    assert gold_rank(docs, 5) == 1
    assert gold_rank(docs, 99) is None
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(None) == 0.0


def main():
    _selftest()

    print(f"Building vector store from {os.path.basename(PDF_PATH)} ...")
    pages = load_pdf(PDF_PATH)
    chunks = chunk_text(pages, source=os.path.basename(PDF_PATH))
    store = create_vector_store(chunks)
    print(f"  {len(pages)} pages -> {len(chunks)} chunks")

    questions = load_or_generate_questions(chunks)

    rows = []
    for i, q in enumerate(questions):
        print(f"[{i + 1}/{len(questions)}] {q['question'][:70]}")
        base = run_arm(q["question"], store, q["gold_page"], rerank=False)
        rer = run_arm(q["question"], store, q["gold_page"], rerank=True)
        base["judge"] = judge(q["question"], q["reference_answer"], base["context"], base["answer"])
        rer["judge"] = judge(q["question"], q["reference_answer"], rer["context"], rer["answer"])
        rows.append({"question": q["question"], "gold_page": q["gold_page"],
                     "baseline": base, "reranked": rer})

    base_sum = summarize(rows, "baseline")
    rer_sum = summarize(rows, "reranked")
    print_report(base_sum, rer_sum)

    # strip non-serializable docs before saving
    for r in rows:
        for arm in ("baseline", "reranked"):
            r[arm].pop("docs", None)
            r[arm].pop("context", None)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"baseline": base_sum, "reranked": rer_sum, "rows": rows}, f, indent=2)
    print(f"\nFull results -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
