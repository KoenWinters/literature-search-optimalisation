A local retrieval-augmented generation (RAG) pipeline for searching a personal collection of documents. Everything runs locally.

I had a large collection of papers which I used for my literature review on FFF process parameters, due to the complexity of these process parameters, and the difficulty
of drawing conclusions from many papers with contradicting results I decided to use this method. 

How it works:
docs/*.pdf
    │  index.py
    V
extract text (PyMuPDF) → repair hyphenation → split into approximately 400-char chunks
    V
embed each chunk (all-MiniLM-L6-v2) → index/vectors.npy + index/chunks.json
    │  search.py
    V
embed the question with the same model
    V
cosine similarity against all chunks → top 5
    V
top chunks + question → local LLM (Ollama) → answer with [n] citations

Two scripts, used separate:

index.py does the expensive work: reading PDFs, chunking, embedding. This only has to be run once, or again whenever the documents or chunk settings are changed.
search.py loads the index from disk and only embeds the question. Fast enough to run repeatedly.

Embeddings are normalised to unit length at encoding time, which means cosine similarity reduces to a plain dot product (chunk_vectors @ question_vector). One matrix multiplication scores all ~10,000 chunks at once.

Setup: pip install -r requirements.txt
Install Ollama and pull a model: ollama pull llama3.2
Put your PDFs in docs/, then build the index: python index.py

Usage: python search.py "how does nozzle temperature affect layer adhesion?"

Example output: 10747 chunks from 30 files

Question: how does nozzle temperature affect layer adhesion?

1. score 0.593  [<paper>.pdf]
   Increasing nozzle temperature reduces material viscosity, improving layer
   adhesion and preventing under-extrusion, though also increases the risk of
   over-extrusion...

2. score 0.555  [<paper>.pdf]
   Decreasing nozzle temperature increases viscosity, potentially leading to
   under-extrusion introducing voids, reducing interlayer bonding...

Answer:
Higher nozzle temperature lowers viscosity, which improves adhesion between
layers [1], while lower temperature raises viscosity and can introduce voids
and weaker interlayer bonding [2]. ...

The prompt instructs the model to answer only from the supplied fragments and to say so when the answer is not there. That constraint matters more than model size: without it, a small model happily fills gaps from its own training data, which defeats the purpose of grounding answers in the corpus.

Why local? The documents i used includes unpublished work. Hosted APIs receive the retrieved fragments with every query; running the embedding model and the LLM locally means the text never leaves the machine.

The model is not perfect, but it does the job while still remaining simple. 
Some limitations which need to be fixed in the future:

Sentence splitting is naive. Splitting on ". " breaks on et al., Fig. 3, and decimals like 0.485. It also strips the periods, so chunks read as run-on text. A regex on [.!?] that preserves the punctuation would be better, though abbreviations remain an open problem.
No overlap between chunks. Context that straddles a chunk boundary is lost. Carrying the last sentence of each chunk into the next would cost some storage and fix a class of retrieval misses.
Reference lists pollute results. A chunk of bibliography entries scores highly on a topical query because the topic appears in cited titles, but it contains no substance. Truncating each document at its references section is the obvious fix.
No conversational memory. Each run is independent. Follow-up questions like "and at lower density?" retrieve nothing useful, because retrieval only sees those four words. Handling this properly needs query rewriting: expand the follow-up into a standalone query using the conversation history, then search.
Corpus-level questions do not work. "Which papers discuss nozzle temperature?" cannot be answered, because the model only sees the top 5 fragments and has no view of the other ~10,700. This is a limitation of the approach, not of the implementation — that question needs a filter over all chunks, not a generative answer.
No retrieval evaluation. Result quality is judged by reading the output. A small set of questions with known-correct source documents would make changes to chunk size or the splitter measurable instead of anecdotal.
