import os
import re
import time
import warnings
from typing import List, Dict

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    PyMuPDFLoader = None

# Keep using the common loader while muting only this known sunset warning
warnings.filterwarnings(
    "ignore",
    message=r"`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

# Load keys from a .env file 
load_dotenv()

# Constants and Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "knust-admission-rag")
SPARSE_INDEX_NAME = os.getenv("SPARSE_INDEX_NAME", "knust-rag-sparse")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
PDF_PATH = "../data/admission_requirement.pdf"
NAMESPACE = "__default__"

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

LIGATURE_FIXES = {
    "Applica on": "Application",
    "creden als": "credentials",
    "Mathema cs": "Mathematics",
    "MathemaNcs": "Mathematics",
    "aser": "after",
    "wriMen": "written",
    "submieng": "submitting",
    "and ck to declare": "and tick to declare",
}

def clean_pdf_text(text: str) -> str:
    """
    Clean raw PDF-extracted text for the KNUST admissions document
    (clean/non-OCR extraction variant).
    """
    # Normalize non-breaking spaces
    text = text.replace("\u00a0", " ")
    for broken, fixed in LIGATURE_FIXES.items():
        text = text.replace(broken, fixed)
        
    text = re.sub(r"\bof s the application\b", "of the application", text)
    text = re.sub(r"(?m)^\d{1,3}(?=\d[\.\)]\s?[A-Z])", "", text)
    text = re.sub(r"(?m)^\d{1,3}(?=\d\s+Faculty\b)", "", text)
    text = re.sub(
        r"(?m)^\d{1,3}(?=(Elective|Core|Entry|Faculty|College|Applicants)\b)",
        "",
        text,
    )
    text = re.sub(r"(?im)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def clean_langchain_pdf_docs(lc_docs: list) -> list:
    """
    Clean LangChain PDF documents by using the manual clean_pdf_text() function.
    """
    cleaned_docs = []
    for doc in lc_docs:
        text = clean_pdf_text(doc.page_content)

        if text:  # Skip pages that become empty after cleaning
            cleaned_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "page_label": f"Page {doc.metadata.get('page', 0) + 1}",
                        "source_type": "pdf",
                    },
                )
            )

    return cleaned_docs


def merge_pdf_documents(lc_docs: list) -> list:
    """
    Merge all PDF pages into a single LangChain Document.
    """
    if not lc_docs:
        return []

    combined_text = "\n\n".join(doc.page_content for doc in lc_docs)

    return [
        Document(
            page_content=combined_text,
            metadata={
                "source": lc_docs[0].metadata.get("source", ""),
                "source_type": "pdf",
                "total_pages": len(lc_docs),
                "title": "KNUST Admission Content",
                "doc_id": "KNUST_admission_requirements_pdf",
            },
        )
    ]


def embed_chunks(
    chunks: List[Dict],
    model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> List[Dict]:
    """
    Embed all chunks in the corpus using batch API calls.
    
    Returns:
        The same list of chunks, each augmented with 'embedding' and 'token_count'
    """
    enc = tiktoken.get_encoding("cl100k_base")
    embedded_chunks = []
    total_tokens = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        texts = [c["text"] for c in batch]

        # Count tokens before sending (for cost transparency)
        batch_tokens = sum(len(enc.encode(t)) for t in texts)
        total_tokens += batch_tokens

        response = openai_client.embeddings.create(
            input=texts,
            model=model
        )

        for chunk, emb_data in zip(batch, response.data):
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"]   = emb_data.embedding
            embedded_chunk["token_count"] = len(enc.encode(chunk["text"]))
            embedded_chunks.append(embedded_chunk)

        print(f"  Batch {batch_start // batch_size + 1}: "
              f"{len(batch)} chunks embedded ({batch_tokens} tokens)")
        time.sleep(0.1)  # Gentle rate limiting

    # Cost estimate: $0.02 per 1M tokens
    cost_usd = (total_tokens / 1000000) * 0.02

    print(f"\nEmbedding complete:")
    print(f"  Chunks embedded: {len(embedded_chunks)}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Estimated cost:  ${cost_usd:.6f} USD")
    print(f"  Dimensions:      {len(embedded_chunks[0]['embedding'])}")

    return embedded_chunks


def create_pinecone_index(
    client: Pinecone,
    index_name: str,
    dimension: int,
    cloud: str = "aws",
    region: str = "us-east-1"
) -> object:
    """
    Create a Pinecone serverless index if it doesn't already exist.
    
    Returns:
        Pinecone Index object ready for upsert and query
    """
    existing = [idx.name for idx in client.list_indexes()]

    if index_name in existing:
        print(f"Index '{index_name}' already exists — connecting to it.")
    else:
        print(f"Creating Pinecone index '{index_name}'...")
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        # Wait for the index to be ready
        while not client.describe_index(index_name).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)
        print("  Index created successfully.")

    index = client.Index(index_name)
    stats = index.describe_index_stats()
    print("\nIndex stats:")
    print(f"  Dimension:    {stats.get('dimension', dimension)}")
    print(f"  Total vectors: {stats.get('total_vector_count', 0)}")
    print("  Metric:       cosine")

    return index


def upsert_to_pinecone_sparse(
    index,
    chunks: List[Dict],
    batch_size: int = 96,   # records API batch limit is smaller than vectors API
) -> Dict:
    total_upserted = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]

        records = [
            {
                "_id":         chunk["chunk_id"],
                "chunk_text":  chunk["text"],
                "title":       chunk.get("title", ""),
            }
            for chunk in batch
        ]

        index.upsert_records(namespace=NAMESPACE, records=records)
        total_upserted += len(records)
        print(f"  Upserted batch {batch_start // batch_size + 1}: "
              f"{len(records)} records (total: {total_upserted})")

    time.sleep(1)
    stats = index.describe_index_stats()
    print("\nSparse index ready:")
    print(f"  Total records stored: {stats.get('total_vector_count', total_upserted)}")
    print(f"  Index name:           {SPARSE_INDEX_NAME}")

    return {"total_upserted": total_upserted}


def main():
    """
    Main execution pipeline. Runs only if the script is executed directly.
    """
    if PyMuPDFLoader is None:
        raise RuntimeError(
            "PyMuPDFLoader not available. Install indexing dependencies: "
            "pip install -e '.[indexing]'"
        )
    
    # 1. Load PDF
    print("Loading PDF...")
    loader = PyMuPDFLoader(PDF_PATH)
    lc_pdf_pages = loader.load()
    print(f"  Documents loaded: {len(lc_pdf_pages)} (one per page)\n")

    # 2. Clean and Merge
    print("Cleaning and merging documents...")
    cleaned_pdf_docs = clean_langchain_pdf_docs(lc_pdf_pages)
    full_doc = merge_pdf_documents(cleaned_pdf_docs)

    # 3. Chunk documents
    print("Chunking documents...")
    production_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunked_docs = production_splitter.split_documents(full_doc)

    # Assign clean chunk IDs for Pinecone
    for i, chunk in enumerate(chunked_docs):
        doc_id = chunk.metadata.get("doc_id", "doc")
        chunk.metadata["chunk_id"] = f"{doc_id}_{i:03d}"
        
    print(f"  Total chunks: {len(chunked_docs)}\n")

    # 4. Embeddings
    print("Embedding KenteCode AI corpus...")
    chunks_as_dicts = [
        {"text": doc.page_content, "chunk_id": f"chunk_{i}", **doc.metadata}
        for i, doc in enumerate(chunked_docs)
    ]
    embedded_chunks = embed_chunks(chunks_as_dicts, model=EMBEDDING_MODEL, batch_size=50)

    # 5. Pinecone Dense Index Initialization
    print("\nInitializing Pinecone Dense Index...")
    pinecone_index = create_pinecone_index(pinecone_client, INDEX_NAME, EMBEDDING_DIM)

    # 6. Pinecone Sparse Index Initialization
    print("\nInitializing Pinecone Sparse Index...")
    if pinecone_client.has_index(SPARSE_INDEX_NAME):
        print(f"Deleting existing index '{SPARSE_INDEX_NAME}'...")
        pinecone_client.delete_index(SPARSE_INDEX_NAME)
        time.sleep(5)

    print(f"Creating '{SPARSE_INDEX_NAME}' with integrated embedding model...")
    pinecone_client.create_index_for_model(
        name=SPARSE_INDEX_NAME,
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1,
        embed={
            "model": "pinecone-sparse-english-v0",
            "field_map": {"text": "chunk_text"},
        },
    )

    while not pinecone_client.describe_index(SPARSE_INDEX_NAME).status["ready"]:
        print("  Waiting for index to be ready...")
        time.sleep(2)
    print("  Index created successfully.")
    
    pinecone_index_sparse = pinecone_client.Index(SPARSE_INDEX_NAME)

    # 7. Upsert Sparse Data
    print("\nUpserting chunks into sparse index...")
    upsert_to_pinecone_sparse(pinecone_index_sparse, embedded_chunks)


if __name__ == "__main__":
    main()