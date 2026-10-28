# Multi-Model RAG Assessment

Two retrieval-augmented question-answering pipelines built on the same retriever and scored on SQuAD. One uses an **extractive BERT reader** and the other a **generative local LLM** (Mistral-NeMo 12B through Ollama), so the comparison isolates what the answering model contributes.

## Pipeline

```
Indexing:  SQuAD contexts → chunk (500 chars, 50 overlap) → embed (all-MiniLM-L6-v2) → ChromaDB

Querying:  question → embed → top-k search in ChromaDB
                                    │
                                    ▼
                re-rank with a cross-encoder (ms-marco-MiniLM-L-12-v2)
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
      BERT-large reader (top 2)          Mistral-NeMo 12B via Ollama (top 3)
      extracts an answer span            writes an answer from the contexts
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    ▼
                         SQuAD Exact Match / F1
```

| Step | Script | What it does |
|---|---|---|
| 0 | `download_squad.py` | Downloads SQuAD v1.1 and writes `train.csv` and `validation.csv` |
| 1 | `processing.py` | De-duplicates contexts, splits them into chunks, embeds them, and stores them in a persistent ChromaDB collection with title and source metadata |
| 2 | `building_RAG_usingBert.py` | Retrieves 15 chunks, re-ranks them, and passes the top 2 to `bert-large-uncased-whole-word-masking-finetuned-squad` |
| 3 | `building_RAG.py` | Retrieves 10 chunks, re-ranks them, and prompts `mistral-nemo:12b` with the top 3 at temperature 0. It cleans up answer prefixes and logs whether each answer appears word for word in the retrieved text |

Steps 2 and 3 both score the first 100 validation questions with the SQuAD Exact Match and F1 metrics.

## Running it

```bash
pip install -r requirements.txt

# 0. Get the data (add --train-limit 5000 for a faster first run)
python download_squad.py

# 1. Build the vector database (creates ./my_local_squad_chroma_db)
python processing.py

# 2. Extractive pipeline
python building_RAG_usingBert.py

# 3. Generative pipeline (needs Ollama running locally)
ollama pull mistral-nemo:12b
python building_RAG.py
```

Run every step from the repository folder, since the scripts use relative paths for the CSVs and the database. Retrieval depth, the number of contexts passed to the reader, and the models used are set as constants at the top of each script.

## Team

Built by [Tejas Mutneja](https://github.com/TejasMutneja) and [Agastya Mukherjee](https://github.com/amukhs1325). Originally developed in [TejasMutneja/Multi-Model-Retrieval-Augmented-Generation-Assessment](https://github.com/TejasMutneja/Multi-Model-Retrieval-Augmented-Generation-Assessment).
