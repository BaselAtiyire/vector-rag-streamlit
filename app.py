import os
import uuid
from typing import List, Optional

import streamlit as st

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "patient_records.csv")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "documents_v2"          # bumped — new embedding model

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

st.set_page_config(page_title="Vector Search + RAG", layout="wide")


# -----------------------------
# IMPROVEMENT 2: Better embedding model
# was: all-MiniLM-L6-v2
# now: BAAI/bge-large-en-v1.5  (MTEB SOTA for retrieval)
# -----------------------------
def make_embedding_function():
    return SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-large-en-v1.5",
        device="cpu",
        normalize_embeddings=True,          # required for BGE cosine similarity
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
        metadata={"hnsw:space": "cosine"},  # cosine distance for BGE
    )


# -----------------------------
# IMPROVEMENT 4: Reranker
# Cross-encoder rescores shortlist → top_n best chunks surfaced
# -----------------------------
@st.cache_resource
def get_reranker():
    if not RERANKER_AVAILABLE:
        return None
    try:
        return CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    except Exception:
        return None


def rerank(query: str, docs: List[str], metas: List[dict], top_n: int = 5):
    reranker = get_reranker()
    if reranker is None or not docs:
        return docs, metas

    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
    ranked = ranked[:top_n]
    return [r[1] for r in ranked], [r[2] for r in ranked]


# -----------------------------
# IMPROVEMENT 5: Tighter, citation-aware prompt
# -----------------------------
def build_context(docs: List[str], metas: List[dict], max_chars: int = 4000) -> str:
    chunks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("filename", "unknown")
        page = meta.get("page", "")
        page_str = f" p.{page}" if page != "" else ""
        chunks.append(f"[Source {i}: {source}{page_str}]\n{doc}")
    return "\n\n".join(chunks)[:max_chars]


def get_groq_api_key() -> Optional[str]:
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return os.getenv("GROQ_API_KEY")


def groq_answer(query: str, context: str) -> str:
    groq_key = get_groq_api_key()
    if not groq_key:
        return (
            "⚠️ GROQ_API_KEY not found. Add it in Streamlit Cloud "
            "(Settings → Secrets) or set it as an environment variable locally."
        )
    if not GROQ_AVAILABLE:
        return "⚠️ langchain-groq not installed. Add `langchain-groq` to requirements.txt."

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,                    # lower = more faithful to context
        groq_api_key=groq_key,
    )

    # IMPROVEMENT 5: Grounded, citation-aware system prompt
    prompt = f"""You are a precise assistant. Follow these rules strictly:
1. Answer ONLY using the numbered sources in the Context section below.
2. After each factual claim, cite the source number in brackets, e.g. [Source 2].
3. If the answer is not found in the context, respond exactly:
   "I don't have enough information in the provided documents."
4. Do not speculate, infer, or use outside knowledge.
5. Be concise: 2–4 sentences unless the question demands more detail.

Question: {query}

Context:
{context}

Answer:""".strip()

    return llm.invoke(prompt).content


# -----------------------------
# Ingestion
# IMPROVEMENT 1: Tuned chunking  (chunk_size↓, overlap↑, natural separators)
# was: chunk_size=900, chunk_overlap=150
# now: chunk_size=512, chunk_overlap=64, explicit separators
# -----------------------------
def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )


def ingest_csv_reset(client) -> Optional[object]:
    if not os.path.exists(DATA_PATH):
        st.error(f"CSV not found at: {DATA_PATH}")
        return None

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection(client)
    loader = CSVLoader(file_path=DATA_PATH)
    docs = loader.load()

    documents = [d.page_content for d in docs]
    metadatas = [
        {"source_type": "csv", "filename": os.path.basename(DATA_PATH)}
        for _ in documents
    ]
    ids = [f"csv_{i}" for i in range(len(documents))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    st.success(f"✅ Reset + ingested {collection.count()} CSV rows.")
    return collection


def ingest_pdfs_append(client, pdf_paths: List[str]) -> Optional[object]:
    if not pdf_paths:
        return None

    collection = get_collection(client)
    splitter = make_splitter()
    all_docs, all_metas, all_ids = [], [], []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        filename = os.path.basename(pdf_path)

        for c in chunks:
            all_docs.append(c.page_content)
            all_metas.append({
                "source_type": "pdf",
                "filename": filename,
                "page": c.metadata.get("page", ""),
            })
            all_ids.append(f"pdf_{uuid.uuid4().hex}")

    if all_docs:
        collection.add(documents=all_docs, metadatas=all_metas, ids=all_ids)
        st.success(f"✅ Appended {len(all_docs)} chunks from {len(pdf_paths)} PDF(s).")
    else:
        st.warning("No text chunks extracted from the uploaded PDF(s).")

    return collection


# -----------------------------
# UI
# -----------------------------
st.title("📄 Vector Search + RAG (CSV + PDF)")
st.caption(
    "Embeddings: `BAAI/bge-large-en-v1.5` · "
    "Reranker: `BAAI/bge-reranker-base` · "
    "LLM: Groq `llama-3.1-8b-instant`"
)

client = get_client()
collection = get_collection(client)

colA, colB = st.columns(2)
with colA:
    st.metric("Vectors in DB", collection.count())
with colB:
    reranker_status = "✅ loaded" if (RERANKER_AVAILABLE and get_reranker()) else "❌ unavailable"
    st.metric("Reranker", reranker_status)

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Ingestion")

    if st.button("📥 Reset DB + Ingest CSV"):
        collection = ingest_csv_reset(client)

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if st.button("📄 Append Uploaded PDFs") and uploaded_files:
        saved_paths = []
        for f in uploaded_files:
            save_path = os.path.join(UPLOADS_DIR, f.name)
            with open(save_path, "wb") as out_f:
                out_f.write(f.read())
            saved_paths.append(save_path)
        collection = ingest_pdfs_append(client, saved_paths)

    st.subheader("Search settings")

    # IMPROVEMENT 3: fetch_k > top_k so reranker has candidates to work with
    top_k = st.slider("Final results (after rerank)", 1, 10, 5)
    fetch_k = st.slider(
        "Candidate pool (before rerank)",
        min_value=top_k,
        max_value=30,
        value=max(top_k * 3, 15),
        help="Fetch this many by vector similarity, then rerank to top_k.",
    )
    source_filter = st.selectbox("Search source", ["All", "CSV only", "PDF only"])
    show_table = st.checkbox("Show table view", value=True)
    use_rerank = st.checkbox(
        "Enable reranking",
        value=RERANKER_AVAILABLE,
        disabled=not RERANKER_AVAILABLE,
    )
    use_groq = st.checkbox("Generate Groq RAG answer", value=False)

with right:
    st.subheader("Ask a question")
    st.caption("Tip: try keywords in your data (e.g., 'hypertension', 'lab result').")
    query = st.text_input("Question", value="hypertension")

    if st.button("🔎 Search"):
        if collection.count() == 0:
            st.warning("DB is empty — ingest some data first.")
            st.stop()

        where = None
        if source_filter == "CSV only":
            where = {"source_type": "csv"}
        elif source_filter == "PDF only":
            where = {"source_type": "pdf"}

        # IMPROVEMENT 3: fetch fetch_k candidates for reranker
        query_kwargs = dict(query_texts=[query], n_results=min(fetch_k, collection.count()))
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        docs  = results["documents"][0]
        metas = results.get("metadatas", [[]])[0]

        # IMPROVEMENT 4: Rerank candidates down to top_k
        if use_rerank and docs:
            with st.spinner("Reranking…"):
                docs, metas = rerank(query, docs, metas, top_n=top_k)
        else:
            docs  = docs[:top_k]
            metas = metas[:top_k]

        st.write(
            f"**Matches:** {len(docs)}  |  "
            f"**Source filter:** {source_filter}  |  "
            f"**Reranked:** {'yes' if use_rerank else 'no'}"
        )

        if show_table and PANDAS_AVAILABLE:
            rows = []
            for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                meta = meta or {}
                preview = doc[:180] + "…" if len(doc) > 180 else doc
                rows.append({
                    "match":       i,
                    "source_type": meta.get("source_type", ""),
                    "filename":    meta.get("filename", ""),
                    "page":        meta.get("page", ""),
                    "preview":     preview,
                })

            st.subheader("📊 Results Table")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.subheader("🔎 Full Text")
            for r in rows:
                with st.expander(
                    f"Match {r['match']} — {r['source_type']} — "
                    f"{r['filename']} — page {r['page']}"
                ):
                    st.code(docs[r["match"] - 1], language="text")
        else:
            for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                with st.expander(f"Match {i} — {meta}"):
                    st.code(doc, language="text")

        if use_groq:
            st.divider()
            st.subheader("🧠 RAG Answer")
            with st.spinner("Generating answer…"):
                context = build_context(docs, metas)
                answer  = groq_answer(query, context)
            st.write(answer)
            st.caption(
                "Answer grounded in reranked matches above. "
                "Source numbers in brackets refer to the matches table."
            )
