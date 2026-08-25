import os
from typing import List, Dict

from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "knust-admission-rag")
SPARSE_INDEX_NAME = os.getenv("SPARSE_INDEX_NAME", "knust-rag-sparse")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Constants
NAMESPACE = "__default__"
RERANK_MODEL = "bge-reranker-v2-m3"
SYSTEM_PROMPT = """You are the KNUST AI Admission assistant. Answer ONLY from the provided 
        context. If the answer is not in the context, say you don't have that
        information. Cite the chunk numbers you used, e.g. [1]."""

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


def connect_to_index(client: Pinecone, index_name: str):
    """Connect to the specified index, raising an error if it does not exist."""
    if not client.has_index(index_name):
        raise ValueError(f"Index '{index_name}' not found.")
    else:
        print(f"Connected to index '{index_name}'.")
    return client.Index(index_name)


# Connect to vector databases
try:
    pinecone_index_dense = connect_to_index(pinecone_client, INDEX_NAME)
    pinecone_index_sparse = connect_to_index(pinecone_client, SPARSE_INDEX_NAME)
except ValueError as e:
    print(f"Warning: {e} - Search functions will fail until indices are created.")
    pinecone_index_dense, pinecone_index_sparse = None, None


def embed_query(text: str) -> List[float]:
    """Embed a single query with the same model used to index the corpus."""
    response = openai_client.embeddings.create(
        input=text, model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def vector_search(query: str, top_k: int = 5, filter_doc_id: str = None) -> List[Dict]:
    """Dense retrieval: embed the query and return the nearest chunks in Pinecone."""
    kwargs = {
        "vector":           embed_query(query),
        "top_k":            top_k,
        "include_metadata": True,
    }
    if filter_doc_id:
        kwargs["filter"] = {"doc_id": {"$eq": filter_doc_id}}

    matches = pinecone_index_dense.query(**kwargs).matches
    return [
        {
            "chunk_id": m.id,
            "score":    m.score,          
            "text":     m.metadata.get("text", ""),
            "title":    m.metadata.get("title", ""),
        }
        for m in matches
    ]


def bm25_search(query: str, top_k: int = 5) -> List[Dict]:
    """Lexical (BM25-style) retrieval using Pinecone's native sparse index."""
    resp = pinecone_index_sparse.search(
        namespace=NAMESPACE,
        query={"top_k": top_k, "inputs": {"text": query}},
        fields=["chunk_text", "title"],
    )    
    return [
        {
            "chunk_id": h["id"],
            "score":    h["score"],       
            "text":     h["fields"].get("chunk_text", ""),
            "title":    h["fields"].get("title", ""),
        }
        for h in resp["result"]["hits"]
    ]


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
    top_k: int = 5,
) -> List[Dict]:
    """Fuse multiple ranked result lists into one, using RRF."""
    scores: Dict[str, float] = {}
    meta:   Dict[str, Dict]  = {}

    for results in result_lists:
        for rank, r in enumerate(results):        
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            meta.setdefault(cid, r)               

    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        {
            "chunk_id":  cid,
            "rrf_score": scores[cid],
            "text":      meta[cid]["text"],
            "title":     meta[cid]["title"],
        }
        for cid in ranked
    ]


def hybrid_search(query: str, top_k: int = 5, candidates: int = 10) -> List[Dict]:
    """Dense + sparse retrieval fused with Reciprocal Rank Fusion."""
    dense  = vector_search(query, top_k=candidates)
    sparse = bm25_search(query, top_k=candidates)
    return reciprocal_rank_fusion([dense, sparse], top_k=top_k)


def cross_encoder_rerank(query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
    """Rerank candidates with Pinecone's hosted reranking API."""
    result = pinecone_client.inference.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=[{"id": str(i), "text": c["text"]} for i, c in enumerate(candidates)],
        top_n=top_k,
        return_documents=False,
    )

    reranked = []
    for hit in result.data:
        c = candidates[int(hit.index)]
        reranked.append({**c, "rerank_score": hit.score})
    return reranked


def rewrite_query(query: str, model: str = CHAT_MODEL) -> str:
    """Turn a messy/conversational question into a clean, SPECIFIC search query
    for the KNUST undergraduate and postgraduate admissions knowledge base.
    """
    prompt = (
        "Rewrite the user's message into a short, specific search query for a "
        "knowledge base about KNUST (Kwame Nkrumah University of Science and "
        "Technology) undergraduate admissions — entry requirements, "
        "programmes, cut-off aggregates, application steps, and fees.\n\n"
        "Rules:\n"
        "- Fix spelling and remove filler words (e.g. \"hey so like\", \"??\").\n"
        "- KEEP every specific detail from the original message — programme "
        "name, subject combos (e.g. WASSCE electives), campus (e.g. Obuasi), "
        "qualification type (WASSCE, A-Level, HND, IB, mature applicant), "
        "numbers, fees, deadlines. Do not replace them with generic phrases.\n"
        "- Do NOT add generic branding like \"KNUST admissions guide\" unless "
        "the message is actually about the university in general.\n"
        "- Keep it under 12 words. Return ONLY the rewritten query, no quotes.\n\n"
        "Examples:\n"
        "User message: hey so like wat grades i need for that computer sci "
        "thing\n"
        "Search query: WASSCE entry requirements for BSc Computer Science\n\n"
        "User message: im a mature applicant no A Level can i still apply for "
        "nursing\n"
        "Search query: mature applicant entry requirements for BSc Nursing\n\n"
        "User message: how much do international students pay??\n"
        "Search query: application processing fee for international applicants\n\n"
        "User message: wats the cutoff for medicine last yr\n"
        "Search query: cut-off aggregate for BSc Human Biology Medicine\n\n"
        "User message: can hnd holders join computer engineering and wat yr\n"
        "Search query: HND holder entry requirements BSc Computer Engineering\n\n"
        f"User message: {query}\nSearch query:"
    )
    resp = openai_client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip().strip('"')


def generate_query_variations(query: str, n: int = 3, model: str = CHAT_MODEL) -> List[str]:
    """Ask the LLM for n alternative phrasings; return the original + variations."""
    prompt = (
        f"Generate {n} alternative phrasings of the question below to improve "
        "search recall over a knowledge base. Vary the vocabulary. "
        "Return each phrasing on its own line, with no numbering or bullets.\n\n"
        f"Question: {query}"
    )
    resp = openai_client.chat.completions.create(
        model=model, temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    variations = [
        line.strip("-•* ").strip()
        for line in resp.choices[0].message.content.splitlines()
        if line.strip()
    ]
    return [query] + variations[:n]


def multi_query_search(query: str, top_k: int = 5, candidates: int = 8):
    """Run hybrid search for several query phrasings and fuse with RRF."""
    variations = generate_query_variations(query)
    result_lists = []
    for v in variations:
        result_lists.append(vector_search(v, top_k=candidates))
        result_lists.append(bm25_search(v, top_k=candidates))
    fused = reciprocal_rank_fusion(result_lists, top_k=top_k)
    return fused, variations


class AdvancedRetriever:
    """Production-style retriever combining multiple techniques.

    Stages (each toggleable):
        1. rewrite       — normalize the raw user question
        2. multi_query   — expand into several phrasings for higher recall
        3. hybrid        — dense (Pinecone) + sparse (BM25) per phrasing
        4. RRF           — fuse all result lists into one ranked pool
        5. rerank        — Pinecone hosted cross-encoder second pass for precision
    """

    def __init__(self, top_k: int = 3, candidates: int = 10):
        self.top_k = top_k
        self.candidates = candidates

    def retrieve(
        self,
        query: str,
        use_rewrite: bool = True,
        use_multi_query: bool = True,
        use_rerank: bool = True,
        verbose: bool = False,
    ) -> List[Dict]:
        trace = {}

        # 1. Rewrite
        search_query = rewrite_query(query) if use_rewrite else query
        trace["search_query"] = search_query

        # 2. Multi-query expansion
        queries = (
            generate_query_variations(search_query, n=2)
            if use_multi_query else [search_query]
        )
        trace["queries"] = queries

        # 3 + 4. Hybrid search per query, fused with RRF
        result_lists = []
        for q in queries:
            result_lists.append(vector_search(q, top_k=self.candidates))
            result_lists.append(bm25_search(q, top_k=self.candidates))
        fused = reciprocal_rank_fusion(result_lists, top_k=self.candidates)

        # 5. Rerank (or just take the fused top-k)
        final = (
            cross_encoder_rerank(query, fused, top_k=self.top_k)
            if use_rerank else fused[:self.top_k]
        )

        if verbose:
            print(f"  rewritten : {trace['search_query']}")
            print(f"  phrasings : {len(trace['queries'])}")
            print(f"  fused pool: {len(fused)} candidates")
        return final


# Instantiate a default retriever for easy importing
default_retriever = AdvancedRetriever(top_k=3, candidates=10)


def answer_question(query: str, model: str = CHAT_MODEL) -> str:
    """Full RAG: advanced retrieval + grounded generation for KNUST admissions."""
    chunks = default_retriever.retrieve(query)
    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))

    # Inject the retrieved context into the system prompt
    system = f"{SYSTEM_PROMPT}\n\nRetrieved context:\n{context}"
    
    resp = openai_client.chat.completions.create(
        model=model, 
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    print("Clients initialized.")
    print(f"  Vector DB index (dense) : {INDEX_NAME}")
    print(f"  Vector DB index (sparse): {SPARSE_INDEX_NAME}")
    print(f"  Embedding model : {EMBEDDING_MODEL} ({EMBEDDING_DIM} dims)")
    print(f"  Chat model      : {CHAT_MODEL}")
    
    if pinecone_index_dense and pinecone_index_sparse:
        dense_count  = pinecone_index_dense.describe_index_stats().get("total_vector_count", 0)
        sparse_count = pinecone_index_sparse.describe_index_stats().get("total_vector_count", 0)
        print(f"  Dense vectors  : {dense_count}")
        print(f"  Sparse records : {sparse_count}")
        
    print(f"cross_encoder_rerank() ready — Pinecone hosted reranker ({RERANK_MODEL})\n")

    test_questions = [
        "what's the requirement for reading computer science?",
        "I want to study something medicine related, what are my options?",
        "can I do architecture with visual art background?",
    ]
    
    for q in test_questions:
        print(f"Q: {q}")
        print(f"A: {answer_question(q)}\n")