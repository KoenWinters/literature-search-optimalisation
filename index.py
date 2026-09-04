from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # pymupdf
import re #used to clean the pdf's 

DOCUMENTS = Path("docs")
INDEX_DIR = Path("index")
CHUNK_CHAR = 400
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_pdf_text(text: str) -> str:
    # PDFs break words across lines with a hyphen ("flex-\nibility"). this can be used to bring them back together.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A single line break inside a paragraph is not a real break, this makes it a space.
    # Double breaks (paragraphs) are kept.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text


def load_documents(directory: Path) -> list[tuple[str, str]]:
    #Returns (filename, full_text) for every .pdf file in the folder "docs"
    files = sorted(directory.glob("*.pdf"))

    documents = []
    for f in files:
        with fitz.open(f) as doc:
            text = "\n".join(page.get_text() for page in doc)
        documents.append((f.name, clean_pdf_text(text)))
    return documents


def split_into_chunks(text: str, size: int = CHUNK_CHAR):
    # Cut text into pieces on sentence ends. Sentences are kept whole so a chunk never starts halfway through one.
    # Returns list[str]

    sentences = text.replace("\n", " ").split(". ")
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < size: #checks whether chunk character exceeds limit
            current = f"{current} {sentence}".strip() #attaches new sentence to the current string
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def main():
    # Reads the files and cuts them into chunks:
    chunks: list[str] = []
    sources: list[str] = []
    for filename, text in load_documents(DOCUMENTS):
        for chunk in split_into_chunks(text):
            chunks.append(chunk)
            sources.append(filename)

    # An embedding model turns every chunk into a vector (all-MiniLM-L6-v2 is the embedding model)
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(chunks, normalize_embeddings=True)

    # Write the index to disk so search.py does not have to redo this every run.
    # .npy keeps the array's shape and dtype; the strings go to JSON separately.
    INDEX_DIR.mkdir(exist_ok=True) #creates the folder if it does not exist yet
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps({"chunks": chunks, "sources": sources}),
        encoding="utf-8",
    )
    print(f"{len(chunks)} chunks from {len(set(sources))} files -> {INDEX_DIR}/")


if __name__ == "__main__":
    main()