import os
import re

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import CSVLoader

# Groq LLM (Generation step for RAG)
from langchain_groq import ChatGroq


# -----------------------------
# Paths (your current setup)
# -----------------------------
BASE_DIR = r"C:\Users\basil\OneDrive\Desktop\AIPROJECT\ai-vector-project"
DATA_PATH = os.path.join(BASE_DIR, "patient_records.csv")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

print("✅ Started main.py")
print("📄 CSV path:", DATA_PATH)
print("📄 CSV exists?:", os.path.exists(DATA_PATH))

# -----------------------------
# Chroma client (persistent)
# -----------------------------
client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)
print("✅ Chroma heartbeat:", client.heartbeat())

# -----------------------------
# Embedding function
# -----------------------------
ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=False
)

# -----------------------------
# Helper: parse metadata from each row text
# -----------------------------
def parse_row(text: str) -> dict:
    """
    Expects CSVLoader output like:
      PatientID: 1062
      Name: Patient_62
      Age: 57
      Diagnosis: Hypertension
      LabResult: 180
    """
    meta: dict = {}

    m = re.search(r"PatientID:\s*(\d+)", text)
    if m:
        meta["patient_id"] = int(m.group(1))

    m = re.search(r"Age:\s*(\d+)", text)
    if m:
        meta["age"] = int(m.group(1))

    m = re.search(r"Diagnosis:\s*([^\n]+)", text)
    if m:
        meta["diagnosis"] = m.group(1).strip()

    m = re.search(r"LabResult:\s*(\d+)", text)
    if m:
        meta["lab_result"] = int(m.group(1))

    return meta

# -----------------------------
# Helper: build LLM context from retrieval results
# -----------------------------
def build_context(query_results: dict, max_chars: int = 4000) -> str:
    docs = query_results["documents"][0]
    metas = query_results.get("metadatas", [[]])[0]

    chunks = []
    for i, doc in enumerate(docs, start=1):
        meta = metas[i - 1] if i - 1 < len(metas) else {}
        chunks.append(f"[Match {i}] metadata={meta}\n{doc}")

    context = "\n\n".join(chunks)
    return context[:max_chars]

# -----------------------------
# Collection (rebuild with metadata)
# -----------------------------
REBUILD_COLLECTION = True  # set to False if you don't want to wipe/re-add

if REBUILD_COLLECTION:
    try:
        client.delete_collection("patients_v1")
        print("🧹 Deleted existing collection: patients_v1")
    except Exception:
        pass

collection = client.get_or_create_collection(
    name="patients_v1",
    embedding_function=ef
)

# -----------------------------
# Load CSV and add (with metadata)
# -----------------------------
loader = CSVLoader(file_path=DATA_PATH)
docs = loader.load()
print("✅ Rows loaded from CSV:", len(docs))

documents = [d.page_content for d in docs]
metadatas = [parse_row(t) for t in documents]
ids = [f"row_{i}" for i in range(len(documents))]

# Add only if empty (after rebuild it will be empty)
if collection.count() == 0:
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("✅ Added rows to Chroma:", collection.count())
else:
    print("ℹ️ Collection already has rows:", collection.count())

# -----------------------------
# Query tests (with filters)
# -----------------------------
query = "what patients have hypertension and high lab results?"

# A) Normal semantic query (no filter)
results = collection.query(query_texts=[query], n_results=3)
print("\n🔎 Query (no filter):", query)
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
    print("-" * 50)
    print(f"{i}. {meta}")
    print(doc)

# B) Filtered query (exact match filter) — only Hypertension
results_hyp = collection.query(
    query_texts=[query],
    n_results=5,
    where={"diagnosis": "Hypertension"}
)
print("\n🔎 Query (where diagnosis=Hypertension):", query)
for i, (doc, meta) in enumerate(zip(results_hyp["documents"][0], results_hyp["metadatas"][0]), start=1):
    print("-" * 50)
    print(f"{i}. {meta}")
    print(doc)

# C) Simple range filtering (retrieve more, then filter in Python)
# Example: age >= 50 among Hypertension results
results_more = collection.query(
    query_texts=[query],
    n_results=20,
    where={"diagnosis": "Hypertension"}
)

filtered_age = []
for doc, meta in zip(results_more["documents"][0], results_more["metadatas"][0]):
    if isinstance(meta, dict) and meta.get("age", 0) >= 50:
        filtered_age.append((doc, meta))

print("\n🔎 Query (Hypertension AND age>=50):", query)
for i, (doc, meta) in enumerate(filtered_age[:5], start=1):
    print("-" * 50)
    print(f"{i}. {meta}")
    print(doc)

# D) Range filter by lab_result (retrieve more, then filter in Python)
LAB_THRESHOLD = 160
results_more_lab = collection.query(
    query_texts=[query],
    n_results=30,
    where={"diagnosis": "Hypertension"}
)

filtered_lab = []
for doc, meta in zip(results_more_lab["documents"][0], results_more_lab["metadatas"][0]):
    if isinstance(meta, dict) and meta.get("lab_result", 0) >= LAB_THRESHOLD:
        filtered_lab.append((doc, meta))

print(f"\n🔎 Query (Hypertension AND lab_result>={LAB_THRESHOLD}):", query)
for i, (doc, meta) in enumerate(filtered_lab[:5], start=1):
    print("-" * 50)
    print(f"{i}. {meta}")
    print(doc)

# E) Exact lookup by patient_id (metadata filter)
PATIENT_ID = 1062
patient_hit = collection.get(
    where={"patient_id": PATIENT_ID},
    include=["documents", "metadatas"]
)

print(f"\n🧾 Patient lookup (patient_id={PATIENT_ID}):")
if patient_hit.get("ids"):
    for doc, meta in zip(patient_hit["documents"], patient_hit["metadatas"]):
        print("-" * 50)
        print(meta)
        print(doc)
else:
    print("No patient found with that patient_id.")

# -----------------------------
# RAG Generation Step (Groq LLM)
# -----------------------------
# Requires: environment variable GROQ_API_KEY set in your terminal.
# Windows CMD (run once):  setx GROQ_API_KEY "YOUR_KEY"
groq_key = os.getenv("GROQ_API_KEY")

print("\n🧠 RAG (Groq) generation step:")
if not groq_key:
    print("⚠️ GROQ_API_KEY not found. Set it first, then rerun.")
else:
    context = build_context(results_hyp)  # use filtered (Hypertension) retrieval as context

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_key
    )

    prompt = f"""You are a careful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say: \"I don't have enough information in the provided records.\".

Question: {query}

Context:
{context}"""

    answer = llm.invoke(prompt).content
    print("\n🧠 LLM Answer:\n", answer)

    print("\n📌 Sources (top matches used as context):")
    for i, meta in enumerate(results_hyp.get("metadatas", [[]])[0], start=1):
        print(f"- Match {i}: {meta}")

print("\n✅ Done")
