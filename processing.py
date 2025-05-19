# step1_load_local_db_improved.py
# Purpose: Load SQuAD-like contexts from local CSVs into a local ChromaDB,
# with improved path handling, metadata storage, and chunking options.

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from tqdm import tqdm
from pathlib import Path # For better path handling
# Optional: For advanced chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("---  Step 1: Load Dataset into Local Vector Database ---")

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
# --- File Paths (using pathlib for robustness) ---
# Assumes the script is in your project folder, and CSVs are also there.
PROJECT_ROOT = Path(__file__).parent # Gets the directory where the script is located
TRAIN_CSV_PATH = PROJECT_ROOT / 'train.csv'
VALIDATION_CSV_PATH = PROJECT_ROOT / 'validation.csv'

# --- ChromaDB Configuration ---
LOCAL_DB_PATH = PROJECT_ROOT / "my_local_squad_chroma_db" # New DB path for this version
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
COLLECTION_NAME = f"squad_contexts_md_{EMBEDDING_MODEL_NAME.replace('/', '_').replace('-', '_')}" # Added _md for metadata

# --- CSV Column Names ---
CONTEXT_COLUMN = 'context'
TITLE_COLUMN = 'title' # We'll use this for metadata
ID_COLUMN = 'id'       # Assuming your CSVs have an 'id' column for questions, useful for unique doc IDs

# --- Chunking Configuration (for optional advanced chunking) ---
CHUNK_SIZE = 500      # Max characters per chunk (if using RecursiveCharacterTextSplitter)
CHUNK_OVERLAP = 50   # Overlap between chunks (if using RecursiveCharacterTextSplitter)
# Set to True to enable advanced chunking, False to use full contexts as chunks.
USE_ADVANCED_CHUNKING = True # <<<<< USER ACTION: Set to True to try advanced chunking

print(f"--- Configuration ---")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Training CSV: {TRAIN_CSV_PATH}")
print(f"Validation CSV: {VALIDATION_CSV_PATH}")
print(f"Local ChromaDB Path: {LOCAL_DB_PATH}")
print(f"ChromaDB Collection Name: {COLLECTION_NAME}")
print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"Using Advanced Chunking: {USE_ADVANCED_CHUNKING}")

# --------------------------------------------------------------------------
# 2. Load Data and Prepare Documents with Metadata
# --------------------------------------------------------------------------
print("\n--- Loading and Preparing Documents with Metadata ---")
all_documents_with_metadata = [] # List to hold dicts: {'text': ..., 'metadata': {...}}

def process_df_for_documents(df, df_name=""):
    processed_docs = []
    if df is None:
        return processed_docs
    
    # Ensure required columns exist
    if CONTEXT_COLUMN not in df.columns:
        print(f"Error: Column '{CONTEXT_COLUMN}' not found in {df_name}. Skipping this file.")
        return processed_docs
    if TITLE_COLUMN not in df.columns:
        print(f"Warning: Column '{TITLE_COLUMN}' not found in {df_name}. Title metadata will be missing for these entries.")
    if ID_COLUMN not in df.columns:
        print(f"Warning: Column '{ID_COLUMN}' not found in {df_name}. A generic source_id will be used.")

    # Use a set to store contexts we've already added to avoid duplicates from the same source file
    # if multiple questions share the exact same paragraph.
    # We'll handle global uniqueness later.
    seen_contexts_in_df = set()
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {df_name}"):
        context = str(row[CONTEXT_COLUMN]) if pd.notna(row[CONTEXT_COLUMN]) else None
        title = str(row[TITLE_COLUMN]) if TITLE_COLUMN in df.columns and pd.notna(row[TITLE_COLUMN]) else "Unknown Title"
        # Use question ID for more unique source tracking, or row index if no ID
        source_doc_id = str(row[ID_COLUMN]) if ID_COLUMN in df.columns and pd.notna(row[ID_COLUMN]) else f"{df_name}_row_{index}"

        if context and context not in seen_contexts_in_df:
            processed_docs.append({
                "original_text": context, # Keep original full context for potential splitting
                "metadata": {
                    "source_file": df_name,
                    "title": title,
                    "source_doc_id": source_doc_id # ID of the question/entry this context came from
                }
            })
            seen_contexts_in_df.add(context)
    return processed_docs

try:
    if not TRAIN_CSV_PATH.exists():
        print(f"ERROR: Training CSV file not found at '{TRAIN_CSV_PATH}'.")
        exit()
    squad_train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Successfully loaded training data from: {TRAIN_CSV_PATH} (Shape: {squad_train_df.shape})")
    all_documents_with_metadata.extend(process_df_for_documents(squad_train_df, "train.csv"))
except Exception as e: print(f"Error loading/processing {TRAIN_CSV_PATH}: {e}"); exit()

try:
    if not VALIDATION_CSV_PATH.exists():
        print(f"ERROR: Validation CSV file not found at '{VALIDATION_CSV_PATH}'.")
        exit()
    squad_validation_df = pd.read_csv(VALIDATION_CSV_PATH)
    print(f"Successfully loaded validation data from: {VALIDATION_CSV_PATH} (Shape: {squad_validation_df.shape})")
    all_documents_with_metadata.extend(process_df_for_documents(squad_validation_df, "validation.csv"))
except Exception as e: print(f"Error loading/processing {VALIDATION_CSV_PATH}: {e}"); exit()

# Ensure global uniqueness of contexts before chunking
unique_original_texts = {} # Store original_text -> first_metadata_encountered
for doc in all_documents_with_metadata:
    if doc["original_text"] not in unique_original_texts:
        unique_original_texts[doc["original_text"]] = doc["metadata"]

documents_to_chunk = [{"original_text": text, "metadata": meta} for text, meta in unique_original_texts.items()]
print(f"Total unique original contexts to process: {len(documents_to_chunk)}")

if not documents_to_chunk:
    print("No documents to process. Exiting.")
    exit()

# --------------------------------------------------------------------------
# 3. Chunking Strategy
# --------------------------------------------------------------------------
print("\n--- Applying Chunking Strategy ---")
final_chunks_with_metadata = [] # List of dicts: {'id': str, 'text': str, 'metadata': dict}
chunk_id_counter = 0

if USE_ADVANCED_CHUNKING:
    print(f"Using RecursiveCharacterTextSplitter with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    for doc_info in tqdm(documents_to_chunk, desc="Chunking documents"):
        original_text = doc_info["original_text"]
        base_metadata = doc_info["metadata"]
        
        chunks = text_splitter.split_text(original_text)
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = base_metadata.copy() # Start with base metadata
            chunk_metadata["chunk_number"] = i + 1
            chunk_metadata["total_chunks_in_doc"] = len(chunks)
            
            final_chunks_with_metadata.append({
                "id": f"chunk_{chunk_id_counter}",
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            chunk_id_counter += 1
else:
    print("Using full contexts as single chunks.")
    for doc_info in tqdm(documents_to_chunk, desc="Preparing full context chunks"):
        final_chunks_with_metadata.append({
            "id": f"chunk_{chunk_id_counter}",
            "text": doc_info["original_text"],
            "metadata": doc_info["metadata"] # Base metadata is enough here
        })
        chunk_id_counter += 1

print(f"Total text chunks to add to DB: {len(final_chunks_with_metadata)}")
if not final_chunks_with_metadata:
    print("No text chunks to process after chunking step. Exiting.")
    exit()

# Prepare lists for ChromaDB
chunk_texts_for_db = [chunk['text'] for chunk in final_chunks_with_metadata]
chunk_ids_for_db = [chunk['id'] for chunk in final_chunks_with_metadata]
chunk_metadatas_for_db = [chunk['metadata'] for chunk in final_chunks_with_metadata]


# --------------------------------------------------------------------------
# 4. Embedding Generation
# --------------------------------------------------------------------------
print("\n--- Generating embeddings ---")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Successfully loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e: print(f"Error loading embedding model: {e}"); exit()

batch_size_embed = 64 # Adjust based on your RAM
all_embeddings_list = []

print(f"Generating embeddings for {len(chunk_texts_for_db)} chunks...")
for i in tqdm(range(0, len(chunk_texts_for_db), batch_size_embed), desc="Generating Embeddings"):
    batch_texts = chunk_texts_for_db[i:i + batch_size_embed]
    embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
    all_embeddings_list.extend(embeddings)

if not all_embeddings_list: print("No embeddings generated. Exiting."); exit()

print(f"Generated {len(all_embeddings_list)} embeddings.")
all_embeddings_np = np.array(all_embeddings_list)

# --------------------------------------------------------------------------
# 5. Load into Local Vector Database (ChromaDB)
# --------------------------------------------------------------------------
print("\n--- Setting up Local Vector Database (ChromaDB) ---")
LOCAL_DB_PATH.mkdir(parents=True, exist_ok=True) # Ensure directory exists

client = chromadb.PersistentClient(path=str(LOCAL_DB_PATH)) # ChromaDB expects path as string
print(f"ChromaDB client initialized. Database will be stored at: {LOCAL_DB_PATH}")

try:
    # Delete collection if it exists for a fresh start (good for development)
    # You might want to comment this out if you want to add to an existing collection later.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it for a fresh start.")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection( # Use create_collection for a fresh one
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Using cosine similarity
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' created successfully.")
except Exception as e:
    print(f"Error setting up ChromaDB collection '{COLLECTION_NAME}': {e}")
    # If create_collection fails after delete, or if get_or_create_collection preferred, adjust logic.
    # For now, assuming create_collection is intended after potential delete.
    exit()

embeddings_for_chroma = all_embeddings_np.tolist()
db_add_batch_size = 2000 # Adjust based on ChromaDB performance and data size

print(f"Adding {len(chunk_texts_for_db)} chunks to ChromaDB collection '{COLLECTION_NAME}'...")
for i in tqdm(range(0, len(chunk_texts_for_db), db_add_batch_size), desc="Adding to ChromaDB"):
    batch_ids = chunk_ids_for_db[i:i + db_add_batch_size]
    batch_embeddings = embeddings_for_chroma[i:i + db_add_batch_size]
    batch_documents = chunk_texts_for_db[i:i + db_add_batch_size]
    batch_metadatas = chunk_metadatas_for_db[i:i + db_add_batch_size]
    try:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas # Adding metadata here
        )
    except Exception as e:
        print(f"Error adding batch starting at index {i} to ChromaDB: {e}")

final_collection_count = collection.count()
print(f"Successfully processed documents. Collection '{COLLECTION_NAME}' now has {final_collection_count} items.")
if final_collection_count != len(chunk_texts_for_db):
     print(f"Warning: Number of items in collection ({final_collection_count}) does not match number of chunks intended ({len(chunk_texts_for_db)}).")

# --------------------------------------------------------------------------
# 6. Test Query (Similarity Search)
# --------------------------------------------------------------------------
if final_collection_count > 0:
    print("\n--- Testing a sample query on local DB ---")
    query_text = "What is the main purpose of the BERT model?"
    print(f"Query: \"{query_text}\"")
    query_embedding = embedding_model.encode([query_text])[0].tolist()
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'metadatas', 'distances'] # Request metadata too
        )
        print("\nTop 3 Results from Local ChromaDB:")
        if results and results['documents'] and results['documents'][0]:
            for i_res, doc_content in enumerate(results['documents'][0]):
                doc_id = results['ids'][0][i_res]
                distance = results['distances'][0][i_res]
                metadata = results['metadatas'][0][i_res] if results['metadatas'] else {}
                print(f"\n  Result {i_res+1}:")
                print(f"    ID: {doc_id}")
                print(f"    Distance: {distance:.4f}")
                print(f"    Metadata: {metadata}")
                print(f"    Document: \"{str(doc_content)[:200]}...\"")
        else: print("No results returned or results format is unexpected.")
    except Exception as e: print(f"Error during ChromaDB query: {e}")
else:
    print("\nNo documents in collection to query.")

print("\n--- Step 1 (Load dataset into local vector database) script finished. ---")