# step4_full_ollama_rag_and_raga_debug_span_check.py
# Purpose: Evaluate Ollama RAG with tweaks to the span-in-context check for debugging low scores.

import os, re, ast, string, traceback, json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb, ollama, evaluate

print("--- Step 4: Full Ollama RAG System Setup, Test, and RAGA Evaluation (Debugging Span Check) ---")

# --------------------------------------------------------------------------
# 1. Configuration – tweakable knobs
# --------------------------------------------------------------------------
# In your Step 4 combined script

# --- Ollama Configuration ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL_NAME = "mistral-nemo:12b"  

# --- ChromaDB Configuration (MUST MATCH YOUR NEW LOCAL STEP 1) ---
LOCAL_DB_PATH = "./my_local_squad_chroma_db" # Or whatever path your NEW Step 1 used (e.g., "./my_local_squad_chroma_db_bge_base")

EMBEDDING_MODEL_NAME_FOR_DB   = "all-MiniLM-L6-v2"  # <<<<< UPDATE THIS
COLLECTION_NAME = f"squad_contexts_md_{EMBEDDING_MODEL_NAME_FOR_DB.replace('/', '_').replace('-', '_')}" # This must match the collection from new Step 1

# --- Embedding Model for Queries (MUST be the same as used in Step 1) ---
EMBEDDING_MODEL_NAME_FOR_QUERY = EMBEDDING_MODEL_NAME_FOR_DB  # <<<<< UPDATE THIS

# --- RAG Configuration ---
TOP_N_INITIAL_RETRIEVAL_OLLAMA = 10 # Or try 15-20 to give bge-reranker-large more options
TOP_N_CONTEXTS_FOR_OLLAMA      = 3   # <<<<< YOUR NEW SETTING

# --- RAGA Configuration ---
CROSS_ENCODER_MODEL_NAME_RAGA = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # <<<<< YOUR NEW RE-RANKER
VALIDATION_CSV_PATH_RAGA = "validation.csv"
GROUND_TRUTH_ANSWERS_COLUMN_RAGA = 'answers'
QA_MODEL_NAME_FOR_METRIC = "distilbert-base-cased-distilled-squad" # Stays for metric loading logic

print(f"--- Configuration Summary ---")
print(f"Ollama Model: {OLLAMA_MODEL_NAME}")
print(f"Embedding model: {EMBEDDING_MODEL_NAME_FOR_DB}")
print(f"Cross‑encoder  : {CROSS_ENCODER_MODEL_NAME_RAGA}")
print(f"Initial Contexts Retrieved: {TOP_N_INITIAL_RETRIEVAL_OLLAMA}")
print(f"Contexts Fed to LLM (post-rerank): {TOP_N_CONTEXTS_FOR_OLLAMA}")
print()

# --------------------------------------------------------------------------
# 2. Init models / clients
# --------------------------------------------------------------------------
ollama_client = None
embedding_model = None
collection_local = None
cross_encoder_model = None
try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    print("↳ Ollama client attempting connection...")
    # Quick check (optional, but good)
    # ollama_client.list() # This would error out if server not running
    print("↳ Ollama client initialized (assuming server is running).")
except Exception as e:
    print(f"FAILED to initialize Ollama client: {e}. Please ensure Ollama is running."); exit()

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME_FOR_QUERY)
    print("↳ Embedding model loaded.")
except Exception as e: print(f"FAILED to load embedding model: {e}"); exit()

try:
    if not os.path.isdir(LOCAL_DB_PATH):
        raise FileNotFoundError(f"ChromaDB path '{LOCAL_DB_PATH}' missing → run Step 1 with 'all-mpnet-base-v2'")
    chroma_client_local = chromadb.PersistentClient(path=LOCAL_DB_PATH)
    collection_local    = chroma_client_local.get_collection(name=COLLECTION_NAME)
    print(f"↳ Chroma collection '{COLLECTION_NAME}' with {collection_local.count()} chunks.")
    if collection_local.count() == 0: print("WARNING: ChromaDB collection is empty!")
except Exception as e: print(f"FAILED to connect to ChromaDB: {e}"); exit()

try:
    cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME_RAGA)
    print("↳ Cross‑encoder loaded.")
except Exception as e:
    print(f"Error loading Cross-encoder: {e}. Re-ranking will be skipped.");
    cross_encoder_model = None # Allow to proceed without re-ranking if it fails
print()

# --------------------------------------------------------------------------
# Helper: simple SQuAD‑style normaliser + span‑containment
# --------------------------------------------------------------------------
PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation) # Define translator once

def _norm(txt: str) -> str:
    if not isinstance(txt, str): return "" # Handle non-string inputs gracefully
    return " ".join(txt.lower().translate(PUNCT_TRANSLATOR).split())

def span_in_context(span: str, ctxs) -> bool:
    if not span or not ctxs: return False
    n_span = _norm(span)
    if not n_span: return False # Empty span after normalization
    return any(n_span in _norm(c) for c in ctxs if c) # Ensure context is not None

# --------------------------------------------------------------------------
# 3. RAG core (with modified span check for debugging)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise question‑answering engine. Answer ONLY with a verbatim span "
    "from the contexts. If absent, reply exactly: 'Information not found in contexts.'"
)
FEW_SHOT_EXAMPLES = """Example 1:\nContext 1: The Normans were a population arising from Norse Viking settlers and indigenous West Franks.\nQuestion: The Normans arose from the intermingling between what two groups?\nExtracted Answer: Norse Viking settlers and indigenous West Franks\n\nExample 2:\nContext 1: The Amazon rainforest is …\nQuestion: What is the primary export of Canada?\nExtracted Answer: Information not found in contexts.\n"""

def get_ollama_rag_answer(question_text: str,
                          top_n_initial=TOP_N_INITIAL_RETRIEVAL_OLLAMA,
                          top_n_final=TOP_N_CONTEXTS_FOR_OLLAMA):
    global embedding_model, collection_local, ollama_client, OLLAMA_MODEL_NAME, cross_encoder_model
    if not all([embedding_model, collection_local, ollama_client]):
        return {"answer": "RAG components not ready.", "source_contexts": []}
    if not question_text.strip():
        return {"answer": "Please provide a question.", "source_contexts": []}

    query_emb = embedding_model.encode([question_text])[0].tolist()
    res = collection_local.query(query_embeddings=[query_emb], n_results=top_n_initial, include=['documents'])
    initial_ctxs = [ctx for ctx in res.get('documents', [[]])[0] if ctx] # Filter None contexts

    if not initial_ctxs:
        return {"answer": "No relevant contexts found initially.", "source_contexts": []}
    
    top_ctxs = initial_ctxs # Default if no re-ranking
    if cross_encoder_model and initial_ctxs:
        pairs   = [[question_text, c] for c in initial_ctxs]
        try:
            scores  = cross_encoder_model.predict(pairs, batch_size=32, show_progress_bar=False)
            top_ctxs = [c for c, _ in sorted(zip(initial_ctxs, scores), key=lambda t: t[1], reverse=True)[:top_n_final]]
        except Exception as e_rerank:
            print(f"Warning: Re-ranking failed ({e_rerank}), using top initial contexts.")
            top_ctxs = initial_ctxs[:top_n_final]
    
    if not top_ctxs: # If top_n_final is 0 or all contexts were filtered
        return {"answer": "No contexts left after re-ranking.", "source_contexts": []}

    ctx_block = "\n\n".join(f"Context {i+1}: {c}" for i, c in enumerate(top_ctxs))
    user_content = f"{FEW_SHOT_EXAMPLES.strip()}\n\n---\n\nContext passages:\n{ctx_block}\n\nQuestion: {question_text}\nExtracted Answer:"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    
    try:
        response = ollama_client.chat(model=OLLAMA_MODEL_NAME, messages=msgs, options={"temperature": 0.0, "top_p": 0.1})
        raw_ans = response['message']['content'].strip()
    except Exception as e_ollama:
        print(f"Error calling Ollama API: {e_ollama}")
        return {"answer": f"Ollama API Error: {e_ollama}", "source_contexts": top_ctxs}

    # Strip common prefixes
    final_ans = raw_ans
    prefixes = [
        "based on the context provided, the answer is", "based on the provided context, the answer is",
        "according to the provided contexts, the answer is", "according to the context provided:",
        "based on the context, the answer is", "according to the context, the answer is",
        "the answer found in the context is", "here's an answer from the context:", # Slight variation
        "from the context, the answer is", "extracted answer:", "the answer is"
    ] # Ensure list is comprehensive
    if final_ans.lower() != "information not found in contexts.": # Don't strip from "not found"
        temp_lower_ans = final_ans.lower()
        for p in prefixes:
            if temp_lower_ans.startswith(p.lower()): # Case-insensitive prefix check
                final_ans = final_ans[len(p):].lstrip(" :").strip()
                break
    
    # --- MODIFIED Span-in-Context Check for Debugging ---
    # We will log when the check *would* fail but let the answer pass for now
    # to see its raw F1/EM score.
    is_verbatim_span_in_context = True # Assume true
    if final_ans.lower() != "information not found in contexts." and final_ans.strip() != "":
        if not span_in_context(final_ans, top_ctxs):
            is_verbatim_span_in_context = False
            # You can add a print here for questions being evaluated if you want to see which ones fail the span check:
            # print(f"DEBUG: QID_placeholder - LLM answer '{final_ans}' NOT a verbatim span in contexts. Question: '{question_text[:50]}...'")
            # To enforce the original stricter rule (which led to low scores):
            # final_ans = "Information not found in contexts."
    # For this run, we are NOT overriding final_ans based on is_verbatim_span_in_context
    # to see the LLM's "raw" (post-prefix-strip) answer's performance.
    # You can add ` "is_verbatim": is_verbatim_span_in_context ` to the return if you want to log it per item.
    
    return {"answer": final_ans, "source_contexts": top_ctxs}

print("Function `get_ollama_rag_answer` (with modified span check) ready.\n")

# --------------------------------------------------------------------------
# 4. Single quick test
# --------------------------------------------------------------------------
_q_test = "What did the Denver Broncos achieve in Super Bowl 50?"
print("Test Q:", _q_test)
test_answer_package = get_ollama_rag_answer(_q_test)
print("Answer :", test_answer_package.get('answer'))
print()

# --------------------------------------------------------------------------
# 5. RAGA Answer‑quality eval
# --------------------------------------------------------------------------
print("\n--- RAGA Component 2: Answer Quality (EM/F1 Score for Ollama RAG) ---")
print(f"Evaluating with Ollama Model: {OLLAMA_MODEL_NAME}, Top {TOP_N_CONTEXTS_FOR_OLLAMA} re-ranked contexts.")

try:
    val_df = pd.read_csv(VALIDATION_CSV_PATH_RAGA)
    # --- Running on 100 samples as requested ---
    NUM_SAMPLES_TO_EVALUATE = 100
    val_df_sample = val_df.head(NUM_SAMPLES_TO_EVALUATE)
    print(f"Loaded {len(val_df)} validation questions, evaluating on {len(val_df_sample)} samples.")

    metric_ver = "squad_v2" if "squad2" in QA_MODEL_NAME_FOR_METRIC.lower() or "v2" in QA_MODEL_NAME_FOR_METRIC.lower() else "squad"
    squad_metric = evaluate.load(metric_ver)
    print(f"SQuAD metric ('{metric_ver}') loaded.")

    preds, refs = [], []
    # Compile regex patterns once before the loop
    pattern_array_with_dtype_re = re.compile(r"array\((?P<content>(?:\[.*?\]|'[^']*'|\"[^\"]*\"|[0-9][0-9\.Ee+-]*))\s*,\s*dtype=[a-zA-Z0-9_]+\)", flags=re.DOTALL)
    pattern_array_no_dtype_re = re.compile(r"array\((?P<content>(?:\[.*?\]|'[^']*'|\"[^\"]*\"|[0-9][0-9\.Ee+-]*))\)", flags=re.DOTALL)

    for index, row in tqdm(val_df_sample.iterrows(), total=len(val_df_sample), desc="RAGA eval"):
        qid, qtxt, gtraw = str(row['id']), str(row['question']), row[GROUND_TRUTH_ANSWERS_COLUMN_RAGA]
        if not qtxt.strip(): continue

        gt_texts_list_val = []
        if isinstance(gtraw, str):
            processed_str_val = gtraw
            try:
                temp_str_val = processed_str_val
                while True:
                    new_temp_str_val = pattern_array_with_dtype_re.sub(r"\g<content>", temp_str_val)
                    if new_temp_str_val == temp_str_val: break
                    temp_str_val = new_temp_str_val
                while True:
                    new_temp_str_val = pattern_array_no_dtype_re.sub(r"\g<content>", temp_str_val)
                    if new_temp_str_val == temp_str_val: break
                    temp_str_val = new_temp_str_val
                processed_str_val = temp_str_val
                answers_data_val = ast.literal_eval(processed_str_val)
                if isinstance(answers_data_val, dict) and 'text' in answers_data_val:
                    ans_field = answers_data_val['text']
                    if isinstance(ans_field, list): gt_texts_list_val = [str(ans).strip() for ans in ans_field if str(ans).strip()]
                    elif isinstance(ans_field, str) and ans_field.strip(): gt_texts_list_val = [ans_field.strip()]
                elif isinstance(answers_data_val, list): gt_texts_list_val = [str(ans).strip() for ans in answers_data_val if str(ans).strip()]
            except Exception as e_parse: print(f"ERROR parsing GT for QID {qid}: {e_parse}. Processed: '{processed_str_val}'")
        elif isinstance(gtraw, dict) and 'text' in gtraw:
            ans_field = gtraw['text']
            if hasattr(ans_field, '__iter__') and not isinstance(ans_field, str): gt_texts_list_val = [str(ans).strip() for ans in ans_field if str(ans).strip()]
            elif isinstance(ans_field, str) and ans_field.strip(): gt_texts_list_val = [ans_field.strip()]
        elif isinstance(gtraw, list): gt_texts_list_val = [str(ans).strip() for ans in gtraw if str(ans).strip()]

        if not gt_texts_list_val:
            print(f"Skipping QID {qid}: no GT answers extracted. Original: {gtraw}")
            continue

        pred_pkg = get_ollama_rag_answer(qtxt)
        predicted_ans = pred_pkg['answer']

        # For detailed failure analysis, you can print here:
        is_em_temp = any(_norm(predicted_ans) == _norm(gt) for gt in gt_texts_list_val)
        # if not is_em_temp: # Or based on some F1 threshold if you calculate it per item
        #     print(f"\nQID: {qid}")
        #     print(f"Q: {qtxt}")
        #     print(f"GT: {gt_texts_list_val}")
        #     print(f"PRED: {predicted_ans}")
        #     print(f"CTX Fed: {pred_pkg.get('source_contexts')}")
        #     print("---")

        preds.append({'id': qid, 'prediction_text': predicted_ans.strip()}) # Ensure prediction is stripped for metric
        # For SQuAD metric, normalization (lowercase, remove punc/articles) is often done by the metric itself.
        # If not, applying _norm to prediction_text and gt_texts_list might be needed for fairer F1.
        # However, for EM, the raw string is usually compared by the metric after its own normalization.
        
        refs.append({'id': qid, 'answers': {'text': gt_texts_list_val, 'answer_start': [-1]*len(gt_texts_list_val)}})

    if preds and refs:
        res = squad_metric.compute(predictions=preds, references=refs)
        print(f"\nExact Match (EM) for {OLLAMA_MODEL_NAME} (on {len(preds)} samples): {res.get('exact_match', res.get('exact'))}")
        print(f"F1 Score for {OLLAMA_MODEL_NAME} (on {len(preds)} samples): {res.get('f1')}")
    else: print("No valid predictions/references for metric computation.")

except FileNotFoundError: print(f"Validation.csv not found – skipping RAGA eval.")
except Exception as e:
    print(f"An unexpected error in RAGA Answer Quality: {type(e).__name__} - {e}")
    traceback.print_exc()

print("\n--- Script finished ---")