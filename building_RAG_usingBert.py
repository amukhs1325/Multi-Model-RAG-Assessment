# step2_bert_rag_local_final.py
# Purpose: Define, test, and evaluate an OPTIMIZED BERT-based RAG system
# with re-ranking, locally in VS Code, with fixes and debugging aids.

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from transformers import pipeline # For the QA model
import torch # For checking GPU availability
import evaluate # Use the new evaluate library
import json
import numpy as np 
from tqdm import tqdm
import ast
import re
import traceback
import time # For timing sections
import string # For punctuation stripping in GT parsing helper
from typing import Any, List, Dict # For type hints

VERBOSE = False # Set to False to silence DEBUG RAG prints

print("--- Step 2: Full BERT RAG System Setup, Test, and RAGA Evaluation (Local Optimized) ---")

# --------------------------------------------------------------------------
# 1. Configuration - KEY AREAS FOR EXPERIMENTATION
# --------------------------------------------------------------------------
# --- ChromaDB Configuration (MUST MATCH YOUR LOCAL STEP 1 SETUP) ---
LOCAL_DB_PATH = "./my_local_squad_chroma_db" # Path to your local ChromaDB from your Step 1 script

# !!! USER ACTION: Set this to the embedding model used to CREATE the DB in Step 1 !!!
# This MUST match your Step 1 script (e.g., the one that had USE_ADVANCED_CHUNKING = True)
EMBEDDING_MODEL_NAME_FOR_DB = 'all-MiniLM-L6-v2' # Matches your last Step 1 script posted
COLLECTION_NAME = f"squad_contexts_md_{EMBEDDING_MODEL_NAME_FOR_DB.replace('/', '_').replace('-', '_')}" # Matches your last Step 1

# --- Embedding Model for Queries (MUST be the same as used in Step 1 for the DB) ---
EMBEDDING_MODEL_NAME_FOR_QUERY = EMBEDDING_MODEL_NAME_FOR_DB

# --- BERT QA Model for the Reader ---
QA_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"

# --- Cross-Encoder for Re-ranking ---
CROSS_ENCODER_MODEL_RERANK = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

# --- Retrieval Parameters ---
TOP_N_INITIAL_RETRIEVAL = 15 # From your last Step 2 script
TOP_N_FOR_QA_MODEL = 2      # From your last Step 2 script

# --- RAGA Configuration ---
VALIDATION_CSV_PATH_RAGA = 'validation.csv'
GROUND_TRUTH_ANSWERS_COLUMN_RAGA = 'answers' # <<<<< CORRECTED VARIABLE NAME TO BE USED

# --- Device Configuration for Debugging ---
FORCED_DEVICE = 'cpu' # Start with 'cpu'. If it works, you can try 'mps' for your MacBook Air.

print(f"--- Configuration (Optimized BERT RAG Local) ---")
print(f"Database Path: {LOCAL_DB_PATH}")
print(f"Collection Name: {COLLECTION_NAME}")
print(f"Embedding Model for DB/Query: {EMBEDDING_MODEL_NAME_FOR_QUERY}")
print(f"QA Model: {QA_MODEL_NAME}")
print(f"Cross-Encoder for Re-ranking: {CROSS_ENCODER_MODEL_RERANK}")
print(f"Initial Contexts Retrieved: {TOP_N_INITIAL_RETRIEVAL}")
print(f"Contexts Fed to BERT QA (post-rerank): {TOP_N_FOR_QA_MODEL}")
print(f"FORCED_DEVICE for models: {FORCED_DEVICE}")


# --------------------------------------------------------------------------
# 2. Initialize Clients and Models
# --------------------------------------------------------------------------
print("\n--- Initializing models and clients ---")
embedding_model = None
collection_local = None
qa_pipeline = None
reranker = None # Changed variable name here to match usage in get_rag_answer_bert
actual_device_used_qa = FORCED_DEVICE

try:
    if VERBOSE: print(f"DEBUG: Loading embedding model: {EMBEDDING_MODEL_NAME_FOR_QUERY} onto device: {FORCED_DEVICE}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME_FOR_QUERY, device=FORCED_DEVICE)
    print(f"Embedding model '{EMBEDDING_MODEL_NAME_FOR_QUERY}' loaded (Device: {FORCED_DEVICE}).")
except Exception as e: print(f"Error loading embedding model: {e}"); exit()

try:
    if VERBOSE: print(f"DEBUG: Connecting to ChromaDB at: {LOCAL_DB_PATH}, Collection: {COLLECTION_NAME}...")
    if not os.path.exists(LOCAL_DB_PATH) or not os.path.isdir(LOCAL_DB_PATH): # Check if dir exists
        print(f"ERROR: Local ChromaDB path '{LOCAL_DB_PATH}' does not exist or is not a directory. Run local Step 1 (with chunking config) first.")
        exit()
    chroma_client_local = chromadb.PersistentClient(path=LOCAL_DB_PATH)
    collection_local = chroma_client_local.get_collection(name=COLLECTION_NAME)
    print(f"Connected to local ChromaDB '{COLLECTION_NAME}' ({collection_local.count()} items).")
    if collection_local.count() == 0: print("WARNING: ChromaDB collection is empty or collection name mismatch.")
except Exception as e: print(f"Error connecting to local ChromaDB: {e}"); exit()

try:
    pipeline_device_arg = -1 
    if FORCED_DEVICE == 'mps':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            pipeline_device_arg = "mps" 
            actual_device_used_qa = "MPS (Apple GPU)"
        else:
            print("Warning: MPS specified but not available/built, defaulting QA pipeline to CPU.")
            actual_device_used_qa = "CPU (MPS fallback)" # Stays CPU if MPS fails
    elif FORCED_DEVICE.startswith("cuda") and torch.cuda.is_available(): # Check for cuda string start
        idx_str = FORCED_DEVICE.split(":")[-1]
        pipeline_device_arg = int(idx_str) if idx_str.isdigit() else 0
        actual_device_used_qa = f"CUDA:{pipeline_device_arg}"
    else: # CPU
        actual_device_used_qa = "CPU"
    
    if VERBOSE: print(f"DEBUG: Attempting to load QA pipeline '{QA_MODEL_NAME}' on device: {actual_device_used_qa} (pipeline arg: {pipeline_device_arg})...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL_NAME, tokenizer=QA_MODEL_NAME, device=pipeline_device_arg)
    print(f"BERT QA pipeline loaded: '{QA_MODEL_NAME}' (Using Device: {actual_device_used_qa}).")
except Exception as e: print(f"Error loading QA pipeline: {e}"); exit()

try:
    if VERBOSE: print(f"DEBUG: Loading Cross-encoder: {CROSS_ENCODER_MODEL_RERANK} onto device: {FORCED_DEVICE}...")
    reranker = CrossEncoder(CROSS_ENCODER_MODEL_RERANK, device=FORCED_DEVICE) 
    print(f"Cross‑encoder '{CROSS_ENCODER_MODEL_RERANK}' loaded (Device: {FORCED_DEVICE}).")
except Exception as e:
    print(f"Error loading Cross-encoder: {e}. Re-ranking will be skipped.")
    # reranker will remain None, get_rag_answer_bert handles this.
print()

# --------------------------------------------------------------------------
# 3. Define IMPROVED `get_rag_answer_bert` function with Re-ranking & Logging
# --------------------------------------------------------------------------
print("\n--- Defining IMPROVED BERT RAG System Core Function (with logging) ---")
def get_rag_answer_bert(question: str, top_n_initial: int = TOP_N_INITIAL_RETRIEVAL, top_n_final: int = TOP_N_FOR_QA_MODEL) -> Dict[str, Any]:
    global embedding_model, collection_local, qa_pipeline, reranker 
    func_start_time = time.time()
    if VERBOSE: print(f"DEBUG RAG: Called for Q: '{question[:70]}...'")

    if not all([embedding_model, collection_local, qa_pipeline]):
        if VERBOSE: print("DEBUG RAG: Essential RAG components not ready.")
        return {"answer": "", "score": 0.0, "source_context_for_best_answer": None, "all_qa_attempts_on_top_contexts": []}
    if not question.strip():
        if VERBOSE: print("DEBUG RAG: Empty question provided.")
        return {"answer": "", "score": 0.0, "source_context_for_best_answer": None, "all_qa_attempts_on_top_contexts": []}
    
    t0 = time.time()
    q_emb = embedding_model.encode([question])[0].tolist()
    if VERBOSE: print(f"DEBUG RAG: Query embedding took {time.time() - t0:.4f}s")
    
    ctxs = []
    try: 
        t0 = time.time()
        res = collection_local.query(query_embeddings=[q_emb], n_results=top_n_initial, include=['documents'])
        ctxs = [c for c in res.get("documents", [[]])[0] if c]
        if VERBOSE: print(f"DEBUG RAG: Initial retrieval ({len(ctxs)} contexts) took {time.time() - t0:.4f}s")
    except Exception as e:
        if VERBOSE: print(f"DEBUG RAG: Error querying ChromaDB: {e}")
        return {"answer": "", "score": 0.0, "source_context_for_best_answer": None, "all_qa_attempts_on_top_contexts": []}
    
    if not ctxs:
        if VERBOSE: print("DEBUG RAG: No relevant contexts found initially.")
        return {"answer": "", "score": 0.0, "source_context_for_best_answer": None, "all_qa_attempts_on_top_contexts": []}

    top_ctxs = ctxs[:top_n_final] 
    if reranker and ctxs: 
        if VERBOSE: print(f"DEBUG RAG: Re-ranking {len(ctxs)} contexts...")
        pairs = [[question, c] for c in ctxs] 
        if pairs:
            t0 = time.time()
            try:
                scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False) 
                scored_contexts = sorted(zip(ctxs, scores), key=lambda t: t[1], reverse=True)
                top_ctxs = [c for c, _ in scored_contexts[:top_n_final]]
                if VERBOSE: print(f"DEBUG RAG: Re-ranking took {time.time() - t0:.4f}s. Top {len(top_ctxs)} selected.")
            except Exception as e_rerank:
                if VERBOSE: print(f"DEBUG RAG: Warning: Error during re-ranking: {e_rerank}. Using top initial contexts.")
        else: 
            if VERBOSE: print("DEBUG RAG: No valid pairs for re-ranking.")
    
    if not top_ctxs:
        if VERBOSE: print("DEBUG RAG: No valid contexts after filtering/re-ranking.")
        return {"answer": "", "score": 0.0, "source_context_for_best_answer": None, "all_qa_attempts_on_top_contexts": []}

    best_answer_text, best_answer_score, best_answer_context_val = None, -1.0, None # Renamed to avoid conflict
    all_qa_attempts_list = []
    if VERBOSE: print(f"DEBUG RAG: Processing {len(top_ctxs)} contexts with QA model '{QA_MODEL_NAME}'...")
    for i, c_text in enumerate(top_ctxs):
        t0 = time.time()
        if VERBOSE: print(f"DEBUG RAG:  Running QA on context {i+1}/{len(top_ctxs)} (len: {len(c_text)} chars)...")
        try:
            out = qa_pipeline(question=question, context=c_text)
            if VERBOSE: print(f"DEBUG RAG:    QA on context {i+1} complete. Answer: '{out['answer']}', Score: {out['score']:.4f}. Time: {time.time() - t0:.4f}s")
            all_qa_attempts_list.append({"context_snippet": c_text[:70]+"...", "answer": out['answer'], "score": out['score']})
            if out["score"] > best_answer_score: # Use local best_answer_score
                best_answer_score, best_answer_text, best_answer_context_val = out['score'], out['answer'], c_text
        except Exception as e_qa:
            if VERBOSE: print(f"DEBUG RAG:    Error during QA for context {i+1}: {e_qa}")
            all_qa_attempts_list.append({"context_snippet": c_text[:70]+"...", "answer": f"QA_ERROR: {e_qa}", "score": 0.0})
    
    if VERBOSE: print(f"DEBUG RAG: All QA processing finished. Total time for RAG function: {time.time() - func_start_time:.4f}s")
    return {
        "answer": best_answer_text if best_answer_text else "Could not extract a confident answer.",
        "score": best_answer_score if best_answer_score > -1.0 else 0.0,
        "source_context_for_best_answer": best_answer_context_val,
        "all_qa_attempts_on_top_contexts": all_qa_attempts_list
    }
print("IMPROVED `get_rag_answer_bert` function (with logging) defined.")

# ---------------------------------------------------------------------------
# Helper to safely parse GT answers (Corrected with regex for array(...))
# ---------------------------------------------------------------------------
print("\n--- Defining Safe Ground Truth Parser ---")
PUNCT_TRANSLATOR_GT = str.maketrans("", "", string.punctuation)

def _norm_gt(txt: str) -> str: 
    if not isinstance(txt, str): return ""
    return " ".join(txt.lower().translate(PUNCT_TRANSLATOR_GT).split())

pattern_array_with_dtype_re = re.compile(r"array\((?P<content>(?:\[.*?\]|'[^']*'|\"[^\"]*\"|[0-9][0-9\.Ee+-]*))\s*,\s*dtype=[a-zA-Z0-9_]+\)", flags=re.DOTALL)
pattern_array_no_dtype_re = re.compile(r"array\((?P<content>(?:\[.*?\]|'[^']*'|\"[^\"]*\"|[0-9][0-9\.Ee+-]*))\)", flags=re.DOTALL)

def safe_parse_gt(raw: Any) -> List[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)): return []
    if isinstance(raw, list): return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, dict) and 'text' in raw:
        field = raw['text']
        # Handle if field itself is a numpy array representation after dict parsing
        if type(field).__module__ == np.__name__ and hasattr(field, 'tolist'):
            field = field.tolist() # Convert numpy array to list
        if isinstance(field, list): return [str(x).strip() for x in field if str(x).strip()]
        return [str(field).strip()] if str(field).strip() else []

    if isinstance(raw, str):
        s = raw.strip()
        try:
            temp_str_val = s
            while True: # Iteratively clean array(...) syntax
                new_temp_str_val = pattern_array_with_dtype_re.sub(r"\g<content>", temp_str_val)
                if new_temp_str_val == temp_str_val: break
                temp_str_val = new_temp_str_val
            while True:
                new_temp_str_val = pattern_array_no_dtype_re.sub(r"\g<content>", temp_str_val)
                if new_temp_str_val == temp_str_val: break
                temp_str_val = new_temp_str_val
            cleaned_s = temp_str_val
        except Exception as e_re:
            if VERBOSE: print(f"DEBUG safe_parse_gt: Regex cleaning failed for '{s}': {e_re}")
            cleaned_s = s 

        try: return safe_parse_gt(json.loads(cleaned_s))
        except (json.JSONDecodeError, TypeError):
            try: return safe_parse_gt(ast.literal_eval(cleaned_s))
            except (ValueError, SyntaxError, TypeError):
                return [cleaned_s] if cleaned_s else []
    return [str(raw).strip()] if str(raw).strip() else []
print("Safe Ground Truth Parser `safe_parse_gt` defined.")

# ---------------------------------------------------------------------------
# 4 ⬩ Sanity test & RAGA Evaluation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Single Test Run of Improved BERT RAG System ---")
    demo_q = "What did the Denver Broncos achieve in Super Bowl 50?"
    print(f"\nTest Question: \"{demo_q}\"")
    
    rag_output_bert_single_test = {}
    if all([embedding_model, collection_local, qa_pipeline]):
        rag_output_bert_single_test = get_rag_answer_bert(demo_q)
        print(f"Generated Answer for Test: {rag_output_bert_single_test.get('answer')}")
    else:
        print("One or more RAG components not initialized. Skipping single test run.")

    print("\n\n--- RAGA - Answer Quality (EM/F1 Score for Improved BERT RAG) ---")
    try:
        if not os.path.exists(VALIDATION_CSV_PATH_RAGA): 
            raise FileNotFoundError(f"Validation CSV not found: {VALIDATION_CSV_PATH_RAGA}")
        validation_df_bert = pd.read_csv(VALIDATION_CSV_PATH_RAGA)
        print(f"Loaded validation data: {len(validation_df_bert)} questions.")

        required_cols_raga = ['id', 'question', GROUND_TRUTH_ANSWERS_COLUMN_RAGA] # Using the corrected var name
        if not all(col in validation_df_bert.columns for col in required_cols_raga):
            raise ValueError(f"Validation CSV missing required columns. Need: {required_cols_raga}. Got: {validation_df_bert.columns.tolist()}")

        metric_version_bert = "squad_v2" if "squad2" in QA_MODEL_NAME.lower() else "squad"
        squad_metric_bert = evaluate.load(metric_version_bert)
        print(f"SQuAD metric ('{metric_version_bert}') loaded.")

        NUM_SAMPLES_TO_EVALUATE = 100 # <<<<< START WITH A VERY SMALL NUMBER (e.g., 1 or 2)
        
        validation_df_sample = validation_df_bert.head(min(NUM_SAMPLES_TO_EVALUATE, len(validation_df_bert)))
        print(f"Running Answer Quality RAGA on {len(validation_df_sample)} samples.")
        
        if len(validation_df_sample) > 0:
            first_row_data_debug = validation_df_sample.iloc[0]
            if VERBOSE:
                print(f"\nDEBUG RAGA: First sample to process - QID: {first_row_data_debug.get('id')}")
                print(f"  Question: {first_row_data_debug.get('question')}")

        if len(validation_df_sample) > 0 and all([embedding_model, collection_local, qa_pipeline]): 
            predictions_for_metric_bert = []
            references_for_metric_bert = []

            for index, row in tqdm(validation_df_sample.iterrows(), total=len(validation_df_sample), desc="BERT RAGA Eval (Local)"):
                q_id, question_text_val, ground_truth_raw = str(row['id']), str(row['question']), row[GROUND_TRUTH_ANSWERS_COLUMN_RAGA] # Using corrected var
                if not question_text_val.strip(): continue

                gt_texts_list_val = safe_parse_gt(ground_truth_raw)

                if not gt_texts_list_val:
                    if VERBOSE: print(f"Skipping QID {q_id} (BERT RAGA): no GT answers extracted. Original: {ground_truth_raw}")
                    continue

                bert_rag_ans_package = get_rag_answer_bert(question_text_val)
                predicted_ans_bert = bert_rag_ans_package.get('answer', "")

                prediction_item_bert = {'id': q_id, 'prediction_text': predicted_ans_bert.strip()}
                if metric_version_bert == "squad_v2":
                    no_answer_prob = 0.0
                    if not predicted_ans_bert.strip() or bert_rag_ans_package.get('score', 0.0) < 0.1:
                         no_answer_prob = 1.0
                    prediction_item_bert['no_answer_probability'] = no_answer_prob
                
                predictions_for_metric_bert.append(prediction_item_bert)
                references_for_metric_bert.append({'id': q_id, 'answers': {'text': [str(t) for t in gt_texts_list_val], 'answer_start': [-1] * len(gt_texts_list_val)}})

            if predictions_for_metric_bert and references_for_metric_bert:
                print("\nComputing SQuAD metrics for OPTIMIZED BERT RAG sample...")
                eval_results_bert = squad_metric_bert.compute(predictions=predictions_for_metric_bert, references=references_for_metric_bert)
                em_bert, f1_bert = eval_results_bert.get('exact_match', eval_results_bert.get('exact')), eval_results_bert.get('f1')
                print(f"Exact Match (EM) for OPTIMIZED BERT RAG (on {len(predictions_for_metric_bert)} samples): {em_bert if em_bert is not None else 'N/A'}")
                print(f"F1 Score for OPTIMIZED BERT RAG (on {len(predictions_for_metric_bert)} samples): {f1_bert if f1_bert is not None else 'N/A'}")
                if 'total' in eval_results_bert: print(f"Total questions evaluated: {eval_results_bert['total']}")
            else: print("No valid data for BERT RAG metric computation.")
        else: print("Validation DataFrame empty, sample size zero, or RAG components not ready.")
    except FileNotFoundError: print(f"Validation CSV '{VALIDATION_CSV_PATH_RAGA}' not found.")
    except Exception as e:
        print(f"An unexpected error in BERT RAGA Answer Quality: {type(e).__name__} - {e}")
        traceback.print_exc()

    print("\n--- Step 2 (Full Optimized BERT RAG with RAGA - Local with Patched GT Parsing & Debug) script finished. ---")