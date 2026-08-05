"""
RAG evaluation harness — measures the cross-encoder reranker's impact.

Runs the REAL pipeline (load_pdf -> chunk -> FAISS -> retrieve -> generate)
twice per question: reranker OFF (baseline) and ON. Reports, per arm:
  Hit@k, MRR, answer correctness (1-5), faithfulness (1-5), retrieval latency.

The store is built from MULTIPLE books, so each question's gold label records
both the source file AND the page — page numbers collide across books, so
retrieval is scored on (source, page), not page alone.

Questions are auto-generated from the PDFs with an LLM (gpt-4o-mini), balanced
across books, deliberately reasoning-heavy (not lookups), and cached to
questions.json so re-runs are deterministic and free.

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
UPLOADS = os.path.join(HERE, "..", "uploads")
PDF_PATHS = [
    os.path.join(UPLOADS, "Grokking_Machine_Learning_-_Luis_Serrano.pdf"),
    os.path.join(UPLOADS, "_OceanofPDF.com_Hands-On_Machine_Learning_with_Scikit-Learn_and_PyTorch_-_Aurelien_Geron.pdf"),
]
QUESTIONS_FILE = os.path.join(HERE, "questions.json")
RESULTS_FILE = os.path.join(HERE, "results.json")

N_PER_BOOK = 20     # questions sampled per book
K = 6               # chunks fed to the LLM
FETCH_K = 20        # candidate pool the reranker re-scores
JUDGE_MODEL = "gpt-4o-mini"
SEED = 42


# --- retrieval metrics (pure) ------------------------------------------------
def gold_rank(retrieved_docs, gold_source, gold_page):
    """1-based rank of the gold (source, page) in the retrieved list, else None."""
    for i, doc in enumerate(retrieved_docs):
        if (doc.metadata.get("source") == gold_source
                and doc.metadata.get("page") == gold_page):
            return i + 1
    return None


def reciprocal_rank(rank):
    return 1.0 / rank if rank else 0.0


# --- shared pipeline setup ---------------------------------------------------
def build_store():
    """Build one FAISS index over every book; return (store, chunks_by_source)."""
    chunks_by_source = {}
    all_chunks = []
    for path in PDF_PATHS:
        name = os.path.basename(path)
        pages = load_pdf(path)
        chunks = chunk_text(pages, source=name)
        chunks_by_source[name] = chunks
        all_chunks.extend(chunks)
        print(f"  {name}: {len(pages)} pages -> {len(chunks)} chunks")
    print(f"Embedding {len(all_chunks)} chunks from {len(PDF_PATHS)} books...")
    return create_vector_store(all_chunks), chunks_by_source


# --- question generation (cached) --------------------------------------------
TOUGH_PROMPT = (
    "You are writing an exam question from a machine-learning textbook passage.\n"
    "Write ONE challenging question that tests real understanding — it must "
    "require reasoning about the concept (why / how / what-would-happen-if / "
    "explain-the-mechanism / apply-the-idea), NOT copying a sentence. It must "
    "still be fully answerable from THIS passage alone. Also give a concise, "
    "correct reference answer.\n"
    'Respond as JSON: {"question": "...", "answer": "..."}\n\nPassage:\n'
)


def generate_questions(chunks_by_source):
    """Sample substantial chunks from EACH book; LLM writes a tough Q + answer."""
    client = _get_client()
    random.seed(SEED)
    questions = []

    for name, chunks in chunks_by_source.items():
        # Longer passages give the model enough material for a reasoning question.
        substantial = [c for c in chunks if len(c["content"]) > 500]
        sample = random.sample(substantial, min(N_PER_BOOK, len(substantial)))
        print(f"Generating {len(sample)} questions from {name}...")

        for c in sample:
            try:
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": TOUGH_PROMPT + c["content"]}],
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
            except Exception as e:
                print(f"  question-gen skipped ({name} p{c['page']}): {e}")
                continue

            if data.get("question") and data.get("answer"):
                questions.append({
                    "question": data["question"].strip(),
                    "reference_answer": data["answer"].strip(),
                    "gold_source": name,
                    "gold_page": c["page"],
                })

    with open(QUESTIONS_FILE, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Generated {len(questions)} questions -> {QUESTIONS_FILE}")
    return questions


def load_or_generate_questions(chunks_by_source):
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE) as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} cached questions from {QUESTIONS_FILE}")
        return questions
    print("No cached questions — generating from the PDFs...")
    return generate_questions(chunks_by_source)


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
def run_arm(question, store, gold_source, gold_page, rerank):
    t0 = time.perf_counter()
    docs = retrieve_chunks(question, store, k=K, fetch_k=FETCH_K, rerank=rerank)
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    rank = gold_rank(docs, gold_source, gold_page)
    context = "\n".join(d.page_content for d in docs)
    answer = generate_answer(question, docs)["answer"]
    return {
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
        def __init__(self, source, page):
            self.metadata = {"source": source, "page": page}
    docs = [D("a.pdf", 5), D("b.pdf", 5), D("a.pdf", 9)]
    assert gold_rank(docs, "b.pdf", 5) == 2      # same page, different book
    assert gold_rank(docs, "a.pdf", 5) == 1
    assert gold_rank(docs, "a.pdf", 99) is None
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(None) == 0.0


def main():
    _selftest()

    print("Building vector store from books...")
    store, chunks_by_source = build_store()

    questions = load_or_generate_questions(chunks_by_source)

    rows = []
    for i, q in enumerate(questions):
        print(f"[{i + 1}/{len(questions)}] ({q['gold_source'][:20]}) {q['question'][:55]}")
        base = run_arm(q["question"], store, q["gold_source"], q["gold_page"], rerank=False)
        rer = run_arm(q["question"], store, q["gold_source"], q["gold_page"], rerank=True)
        base["judge"] = judge(q["question"], q["reference_answer"], base["context"], base["answer"])
        rer["judge"] = judge(q["question"], q["reference_answer"], rer["context"], rer["answer"])
        rows.append({"question": q["question"], "gold_source": q["gold_source"],
                     "gold_page": q["gold_page"], "baseline": base, "reranked": rer})

    base_sum = summarize(rows, "baseline")
    rer_sum = summarize(rows, "reranked")
    print_report(base_sum, rer_sum)

    for r in rows:                       # strip bulky context before saving
        for arm in ("baseline", "reranked"):
            r[arm].pop("context", None)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"baseline": base_sum, "reranked": rer_sum, "rows": rows}, f, indent=2)
    print(f"\nFull results -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
