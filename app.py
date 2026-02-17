import os
import re
import uuid
from typing import List, Optional

import streamlit as st

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional: table view
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

# Optional Groq (RAG generation)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
BASE_DIR = r"C:\Users\basil\OneDrive\Desktop\AIPROJECT\ai-vector-project"
DATA_PATH = os.path.join(BASE_DIR, "patient_records.csv")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "documents_v1"

os.makedirs(UPLOADS_DIR, exist_ok=True)

st.set_page_config(page_title="Vector Search + RAG (CSV + PDF)", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def make_embedding_function():
    return SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=False,
    )


@st.cache_resource
def get_client():
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client):
    ef = make_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )


def build_context(query_results: dict, max_chars: int = 4000) -> str:
    docs = query_results["documents"][0]
    metas = query_results.get("metadatas", [[]])[0]

    chunks = []
    for i, doc in enumerate(docs, start=1):
        meta = metas[i - 1] if i - 1 < len(metas) else {}
        chunks.append(f"[Match {i}] metadata={meta}\n{doc}")

    context = "\n\n".join(chunks)
    return context[:max_chars]


def groq_answer(query: str, context: str) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "GROQ_API_KEY not found. Set it to enable RAG answer."

    if not GROQ_AVAILABLE:
        return "langchain-groq not installed. Install it with: pip install -U langchain-groq"

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_key,
    )

    prompt = f"""You are a careful assistant.
Use ONLY the context below to answer.
If the answer is not in the context, say: "I don't have enough information in the provided documents."

Question: {query}

Context:
{context}
""".strip()

    return llm.invoke(prompt).content


# -----------------------------
# Ingestion
# -----------------------------
def ingest_csv_reset(client) -> Optional[object]:
    """Deletes the collection and re-ingests ONLY the CSV."""
    if not os.path.exists(DATA_PATH):
        st.error(f"CSV not found at: {DATA_PATH}")
        return None

    # Reset collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection(client)

    loader = CSVLoader(file_path=DATA_PATH)
    docs = loader.load()

    documents = [d.page_content for d in docs]
    metadatas = [{"source_type": "csv", "filename": os.path.basename(DATA_PATH)} for _ in documents]
    ids = [f"csv_{i}" for i in range(len(documents))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    st.success(f"Reset + ingested {collection.count()} CSV rows.")
    return collection


def ingest_pdfs_append(client, pdf_paths: List[str]) -> Optional[object]:
    """Appends PDF chunks to the existing collection."""
    if not pdf_paths:
        return None

    collection = get_collection(client)

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    all_docs, all_metas, all_ids = [], [], []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        filename = os.path.basename(pdf_path)

        for c in chunks:
            all_docs.append(c.page_content)
            all_metas.append(
                {
                    "source_type": "pdf",
                    "filename": filename,
                    "page": c.metadata.get("page", None),
                }
            )
            all_ids.append(f"pdf_{uuid.uuid4().hex}")

    if all_docs:
        collection.add(documents=all_docs, metadatas=all_metas, ids=all_ids)
        st.success(f"Appended {len(all_docs)} chunks from {len(pdf_paths)} PDF(s).")
    else:
        st.warning("No text chunks were extracted from the uploaded PDF(s).")

    return collection


# -----------------------------
# UI
# -----------------------------
st.title("📄 Vector Search + RAG (CSV + PDF Upload)")
st.caption("Upload PDFs, embed them, and ask questions. Demo data only.")

client = get_client()
collection = get_collection(client)

colA, colB = st.columns(2)
with colA:
    st.metric("Vectors in DB", collection.count())
with colB:
    st.code(f"Chroma: {CHROMA_PATH}", language="text")

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Ingestion")

    if st.button("📥 Reset DB + Ingest CSV"):
        collection = ingest_csv_reset(client)

    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    if st.button("📄 Append Uploaded PDFs") and uploaded_files:
        saved_paths = []
        for f in uploaded_files:
            save_path = os.path.join(UPLOADS_DIR, f.name)
            with open(save_path, "wb") as out_f:
                out_f.write(f.read())
            saved_paths.append(save_path)

        collection = ingest_pdfs_append(client, saved_paths)

    st.subheader("Search settings")
    top_k = st.slider("Top K results", 1, 10, 5)
    source_filter = st.selectbox("Search source", ["All", "CSV only", "PDF only"])
    show_table = st.checkbox("Show table view", value=True)
    use_groq = st.checkbox("Generate Groq RAG answer", value=False)

with right:
    st.subheader("Ask a question")

    st.caption("Tip: for blood pressure, try: 'hypertension', 'high blood pressure', 'hypotension', 'low blood pressure'.")
    query = st.text_input("Question", value="hypertension")

    if st.button("🔎 Search"):
        where = None
        if source_filter == "CSV only":
            where = {"source_type": "csv"}
        elif source_filter == "PDF only":
            where = {"source_type": "pdf"}

        if where:
            results = collection.query(query_texts=[query], n_results=top_k, where=where)
        else:
            results = collection.query(query_texts=[query], n_results=top_k)

        docs = results["documents"][0]
        metas = results.get("metadatas", [[]])[0]

        st.write(f"**Matches:** {len(docs)}  |  **Source filter:** {source_filter}")

        # -------- Table view --------
        if show_table:
            if not PANDAS_AVAILABLE:
                st.warning("pandas is not installed. Install with: pip install -U pandas (then restart Streamlit).")
            else:
                rows = []
                for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                    meta = meta or {}
                    preview = doc
                    if isinstance(doc, str) and len(doc) > 180:
                        preview = doc[:180] + "…"

                    rows.append(
                        {
                            "match": i,
                            "source_type": meta.get("source_type", ""),
                            "filename": meta.get("filename", ""),
                            "page": meta.get("page", ""),
                            "preview": preview,
                        }
                    )

                df = pd.DataFrame(rows)
                st.subheader("📊 Results Table")
                st.dataframe(df, use_container_width=True)

                st.subheader("🔎 Full Text (expand)")
                for r in rows:
                    idx = r["match"] - 1
                    with st.expander(
                        f"Match {r['match']} — {r['source_type']} — {r['filename']} — page {r['page']}"
                    ):
                        st.write(
                            "**Metadata:**",
                            {"source_type": r["source_type"], "filename": r["filename"], "page": r["page"]},
                        )
                        st.code(docs[idx], language="text")
        else:
            # -------- Classic expandable list --------
            for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                label = f"Match {i} — {meta}"
                with st.expander(label):
                    st.code(doc, language="text")

        if use_groq:
            st.divider()
            st.subheader("🧠 RAG Answer")
            context = build_context(results)
            answer = groq_answer(query, context)
            st.write(answer)
            st.caption("The answer is grounded in the matches above (used as context).")
