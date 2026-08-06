"""RAG evaluation harness: retrieval recall + faithfulness + correctness."""
import os
import sys
import json
import time

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
# load the OpenAI key from backend/.env
load_dotenv(os.path.join(HERE, "..", ".env"))
# let "from backend.app..." imports work when run from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.app.pdf_loader import load_pdf
from backend.app.chunker import chunk_text
from backend.app.rag import create_vector_store, retrieve_chunks
from backend.app.generator import generate_answer
from openai import OpenAI

CORPUS = os.path.join(HERE, "corpus", "ISLP_website.pdf")
EVAL_SET = os.path.join(HERE, "eval_set.json")
INDEX_DIR = os.path.join(HERE, "faiss_index")   # cached vector store
K = 6
JUDGE_MODEL = "gpt-4o-mini"

client = OpenAI()


def get_vector_store():
    """Build the vector store once, then cache it so reruns are fast."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.isdir(INDEX_DIR):
        print("Loading cached FAISS index...")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Building vector store from corpus (first run — a few minutes)...")
    chunks = chunk_text(load_pdf(CORPUS))
    vs = create_vector_store(chunks)
    vs.save_local(INDEX_DIR)
    return vs


def judge_yes_no(prompt):
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().lower().startswith("y")


def judge_faithfulness(answer, context):
    return judge_yes_no(
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Is the Answer fully supported by the Context, with no invented or "
        'contradictory facts? Reply only "yes" or "no".'
    )


def judge_correctness(question, answer, ground_truth):
    return judge_yes_no(
        f"Question: {question}\nReference answer: {ground_truth}\n"
        f"Generated answer: {answer}\n\n"
        "Does the Generated answer convey the same key facts as the Reference "
        'answer? Ignore wording/formatting. Reply only "yes" or "no".'
    )


def main():
    eval_set = json.load(open(EVAL_SET))
    vs = get_vector_store()

    recall = faith = correct = 0
    total_latency = 0.0

    for item in eval_set:
        q = item["question"]
        gold = set(item["relevant_pages"])

        docs = retrieve_chunks(q, vs, k=K)
        got_pages = {d.metadata.get("page") for d in docs}
        r = int(bool(gold & got_pages))

        t0 = time.time()
        result = generate_answer(q, docs)
        latency = time.time() - t0

        answer = result["answer"]
        context = "\n".join(d.page_content for d in docs)
        f = int(judge_faithfulness(answer, context))
        c = int(judge_correctness(q, answer, item["ground_truth_answer"]))

        recall += r; faith += f; correct += c; total_latency += latency
        print(f"Q{item['id']}: recall={r} faithful={f} correct={c}  ({latency:.2f}s)")

    n = len(eval_set)
    print("\n=== SUMMARY ===")
    print(f"Retrieval Recall@{K}: {recall}/{n} = {recall/n:.0%}")
    print(f"Faithfulness:        {faith}/{n} = {faith/n:.0%}")
    print(f"Correctness:         {correct}/{n} = {correct/n:.0%}")
    print(f"Avg latency:         {total_latency/n:.2f}s")


if __name__ == "__main__":
    main()