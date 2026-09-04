import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from index import INDEX_DIR, MODEL_NAME

TOP_K = 5


def load_index():
    # Reads the vectors and chunks that index.py wrote to disk.
    # Returns (vectors, chunks, sources)
    vectors = np.load(INDEX_DIR / "vectors.npy")
    data = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return vectors, data["chunks"], data["sources"]


def answer(question: str, contexts: list[str], sources: list[str]) -> str:
    context_block = "\n\n".join(
        f"[{i}] ({src})\n{text}"
        for i, (text, src) in enumerate(zip(contexts, sources), start=1)
    )

    prompt = f"""Answer the question only on the basis of the fragments underneath. If there is no answer, please respond that you do not know.
    Reference the fragments you are using with [number]. Fragments:
{context_block}

Question: {question}"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
    )
    return response.json()["response"]


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python search.py "your question"')
    question = sys.argv[1]

    # 1. Load the index instead of rebuilding it every run.
    chunk_vectors, chunks, sources = load_index()
    print(f"{len(chunks)} chunks from {len(set(sources))} files")

    # 2. Turns the question into a vector using the same model as index.py.
    model = SentenceTransformer(MODEL_NAME)
    question_vector = model.encode([question], normalize_embeddings=True)[0]

    # 3. Cosine similarity. Because the vectors were normalised to length 1
    scores = chunk_vectors @ question_vector

    # 4. Show the best matches. argsort is ascending, tail is taken and reversed
    best = np.argsort(scores)[-TOP_K:][::-1]

    print(f'\nQuestion: {question}\n')
    for rank, index in enumerate(best, start=1):
        preview = chunks[index][:220].replace("\n", " ")
        print(f"{rank}. score {scores[index]:.3f}  [{sources[index]}]")
        print(f"   {preview}...\n")

    top_chunks = [chunks[i] for i in best]
    top_sources = [sources[i] for i in best]
    print("Answer:")
    print(answer(question, top_chunks, top_sources))


if __name__ == "__main__":
    main()