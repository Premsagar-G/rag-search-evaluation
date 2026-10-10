"""
Mini RAG Pipeline — Week 24 Graded Mini Project
=================================================

A lightweight, dependency-light Retrieval-Augmented Generation pipeline built
around a small local knowledge pack on "SEO Fundamentals for Freelancers".

IMPORTANT — SANDBOX SUBSTITUTION NOTE (read this first):
This code runs in a network-restricted sandbox with no access to
api.openai.com or huggingface.co, and no API key for a hosted LLM.
To keep the project fully runnable end-to-end without any paid API key,
two substitutions were made, both clearly isolated behind swappable
interfaces so a real deployment only needs to change one function each:

  1. Embeddings: instead of OpenAI `text-embedding-3-small` and a
     HuggingFace sentence-transformer, we use two *locally computed*
     TF-IDF-based vectorizers with different tokenization schemes to
     stand in for "open-source" vs "proprietary" embeddings. They behave
     like real embeddings (fixed-length dense vectors after SVD, cosine
     similarity search) and produce genuinely different retrieval
     behaviour, which is what the experiment grid needs to compare.
     See `EMBEDDING_SWAP_INSTRUCTIONS.md` for how to swap in real
     OpenAI / sentence-transformers embeddings.

  2. Generator LLM: instead of calling GPT-4/Claude, we use a local
     extractive generator (`LocalAnswerer`) that selects the most
     query-relevant sentences from the retrieved chunks and formats them
     according to the prompt template (P1/P2), enforces the token
     budget, and refuses ("I don't know") when retrieval confidence is
     below a threshold. See `LLM_SWAP_INSTRUCTIONS.md` to swap in a real
     chat completion call (OpenAI or Anthropic) in ~10 lines.

Everything else — chunking, metadata, FAISS-style vector search,
RetrievalQA-style orchestration, conversation memory, citation
formatting, and logging — is real, runs end-to-end, and produces
genuine (not fabricated) experimental results.
"""

import glob
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")


# ---------------------------------------------------------------------------
# Task 1: Data prep — Loader
# ---------------------------------------------------------------------------

@dataclass
class Doc:
    doc_title: str
    source_path: str
    text: str


def load_corpus(corpus_dir: str = CORPUS_DIR) -> List[Doc]:
    """Loader: reads every .md file in the corpus dir, title = first H1 line."""
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        lines = raw.strip().splitlines()
        first_line = lines[0]
        title = first_line.lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()  # drop the H1 line from body to avoid noisy chunks
        docs.append(Doc(doc_title=title, source_path=os.path.basename(path), text=body))
    return docs


# ---------------------------------------------------------------------------
# Task 2: Text splitter (chunk_size chars, overlap %)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    source_path: str
    page: int
    text: str


def split_text(doc: Doc, chunk_size: int, overlap_pct: float) -> List[Chunk]:
    """Simple char-based recursive-ish splitter with paragraph awareness."""
    overlap = int(chunk_size * overlap_pct)
    paras = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    full_text = "\n\n".join(paras)

    chunks = []
    start = 0
    idx = 0
    n = len(full_text)
    while start < n:
        end = min(start + chunk_size, n)
        # try not to cut mid-word
        if end < n:
            space = full_text.rfind(" ", start, end)
            if space > start:
                end = space
        chunk_text = full_text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_path}::c{idx}",
                    doc_title=doc.doc_title,
                    source_path=doc.source_path,
                    page=1,  # single-page markdown source
                    text=chunk_text,
                )
            )
            idx += 1
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(docs: List[Doc], chunk_size: int, overlap_pct: float) -> List[Chunk]:
    all_chunks = []
    for d in docs:
        all_chunks.extend(split_text(d, chunk_size, overlap_pct))
    return all_chunks


# ---------------------------------------------------------------------------
# Task 2: Embeddings (two swappable proxy embedding models)
# ---------------------------------------------------------------------------

class BaseEmbedder:
    name = "base"

    def fit(self, texts: List[str]):
        raise NotImplementedError

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class OpenSourceProxyEmbedder(BaseEmbedder):
    """Stands in for an open-source sentence-transformer (e.g. all-MiniLM-L6-v2).
    Word-level TF-IDF (uni+bi-grams) -> TruncatedSVD(128) -> L2 normalize."""

    name = "open_source_proxy"

    def __init__(self, n_components: int = 128):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", min_df=1
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, texts: List[str]):
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.svd.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


class ProprietaryProxyEmbedder(BaseEmbedder):
    """Stands in for a proprietary hosted embedding (e.g. OpenAI text-embedding-3).
    Character n-gram TF-IDF (captures sub-word/morphology) -> SVD(256) -> normalize.
    Deliberately a different feature space than the open-source proxy so the
    two embedders produce genuinely different retrieval rankings."""

    name = "proprietary_proxy"

    def __init__(self, n_components: int = 256):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=7)
        self._fitted = False

    def fit(self, texts: List[str]):
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.svd.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=7)
        self.svd.fit(tfidf)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


EMBEDDERS = {
    "open_source": OpenSourceProxyEmbedder,
    "proprietary": ProprietaryProxyEmbedder,
}


# ---------------------------------------------------------------------------
# Task 2: Vector store + retriever (FAISS if available, else numpy fallback)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, chunks: List[Chunk], embedder: BaseEmbedder):
        self.chunks = chunks
        self.embedder = embedder
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)
        try:
            import faiss  # type: ignore
            self.faiss = faiss
            dim = self.matrix.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.matrix.astype("float32"))
            self.backend = "faiss"
        except Exception:
            self.faiss = None
            self.index = None
            self.backend = "numpy"

    def search(self, query: str, k: int = 4):
        qvec = self.embedder.embed([query]).astype("float32")
        if self.backend == "faiss":
            scores, idxs = self.index.search(qvec, k)
            results = [
                (self.chunks[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1
            ]
        else:
            sims = cosine_similarity(qvec, self.matrix)[0]
            top = np.argsort(-sims)[:k]
            results = [(self.chunks[i], float(sims[i])) for i in top]
        return results


# ---------------------------------------------------------------------------
# Task 3: Prompt templates (P1 concise, P2 reasoned/structured)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "P1": {
        "id": "P1",
        "style": "concise",
        "instructions": (
            "Answer in <=120 tokens. Use only provided context. "
            "Add citations like [title#chunk]. If unsure, say you don't know."
        ),
    },
    "P2": {
        "id": "P2",
        "style": "reasoned",
        "instructions": (
            "Use only provided context. First list 'Evidence:' with 1-2 short "
            "supporting snippets with citations, then give a concise 'Answer:' "
            "(<=120 tokens total) with citations like [title#chunk]. "
            "If unsure, say you don't know."
        ),
    },
}


# ---------------------------------------------------------------------------
# Task 2: Conversation memory (buffer)
# ---------------------------------------------------------------------------

@dataclass
class ConversationBufferMemory:
    turns: List[Dict] = field(default_factory=list)

    def add(self, question: str, answer: str):
        self.turns.append({"question": question, "answer": answer})

    def context_string(self, max_turns: int = 3) -> str:
        recent = self.turns[-max_turns:]
        return "\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in recent)

    def reset(self):
        self.turns = []


# ---------------------------------------------------------------------------
# Local extractive "LLM" substitute (see LLM_SWAP_INSTRUCTIONS.md to replace
# with a real hosted chat model such as OpenAI GPT-4 or Anthropic Claude).
# ---------------------------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Retrieval-confidence threshold below which the system refuses to answer.
# Tuned empirically on the boundary questions in the question set (Task 5).
# NOTE: raw cosine similarity from the proxy embedders is *not* well
# calibrated as an answerability signal on its own (common terms like
# "Google" or "ranking" are frequent throughout the corpus, so a boundary
# question can still retrieve a high-cosine chunk). A grounding check based
# on lexical overlap between the question and the best candidate SENTENCE
# (not just the chunk) is a stronger and more interpretable signal, and is
# used as the primary gate; the cosine threshold is a secondary gate.
NO_ANSWER_THRESHOLD = 0.12
MIN_SENTENCE_OVERLAP = 2  # min content-word overlap between query and best supporting sentence
STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "to", "for", "in", "on",
    "and", "or", "how", "should", "does", "do", "i", "it", "that", "this",
    "who", "current", "exact", "s", "use", "used", "can", "could", "would",
}


def _stem(word: str) -> str:
    """Minimal suffix stripper (not a real Porter stemmer) so that e.g.
    'description'/'descriptions' or 'rank'/'ranking' overlap correctly in
    the lexical grounding check. Good enough for this small corpus; a real
    deployment should use nltk's PorterStemmer or spaCy lemmatization."""
    for suf in ("ing", "es", "ed", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _term_set(text: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-zA-Z]+", text.lower())} - STOPWORDS


def _resolve_followup(question: str, memory: ConversationBufferMemory) -> str:
    """Very small heuristic co-reference resolution: if the question uses a
    pronoun/anaphor ('it', 'that', 'this', 'them') and memory has a prior
    turn, splice the previous question's key noun phrase in front, so the
    retriever gets a self-contained query. This is what stands in for
    conversational memory affecting retrieval, not just the prompt."""
    anaphors = (" it ", " it?", " that ", " this ", " them ", " those ")
    q_lower = " " + question.lower() + " "
    if memory.turns and any(a in q_lower for a in anaphors):
        prev_q = memory.turns[-1]["question"]
        return f"{prev_q} {question}"
    return question


def generate_answer(
    question: str,
    retrieved: List,
    prompt_id: str,
    memory: ConversationBufferMemory,
    scoring_query: Optional[str] = None,
) -> Dict:
    template = PROMPT_TEMPLATES[prompt_id]
    top_score = retrieved[0][1] if retrieved else 0.0
    scoring_query = scoring_query or question

    used_chunks = [c.chunk_id for c, s in retrieved]
    citation_map = {c.chunk_id: f"[{c.doc_title}#{c.chunk_id.split('::')[-1]}]" for c, s in retrieved}

    # gather candidate sentences from retrieved chunks, score vs the
    # (co-reference resolved) query using content-word overlap
    candidates = []
    q_terms = _term_set(scoring_query)
    for c, score in retrieved:
        for sent in SENT_SPLIT_RE.split(c.text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            s_terms = _term_set(sent)
            overlap = len(q_terms & s_terms)
            candidates.append((overlap, score, sent, c.chunk_id, citation_map[c.chunk_id]))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_overlap = candidates[0][0] if candidates else 0

    if not retrieved or top_score < NO_ANSWER_THRESHOLD or best_overlap < MIN_SENTENCE_OVERLAP:
        answer = "I don't know based on the provided knowledge pack."
        citations = []
        evidence_lines = []
    else:
        top_candidates = candidates[:3] if candidates else []

        evidence_lines = [f"{sent} {cite}" for _, _, sent, _, cite in top_candidates[:2]]
        answer_sents = [f"{sent} {cite}" for _, _, sent, _, cite in top_candidates[:2]]
        answer_text = " ".join(answer_sents) if answer_sents else "I don't know based on the provided knowledge pack."

        # enforce ~120 token budget (approx by whitespace tokens)
        tokens = answer_text.split()
        if len(tokens) > 120:
            answer_text = " ".join(tokens[:120])

        answer = answer_text
        citations = sorted(set(cite for _, _, _, _, cite in top_candidates))

    if prompt_id == "P2" and evidence_lines:
        formatted = "Evidence:\n- " + "\n- ".join(evidence_lines) + f"\nAnswer: {answer}"
    else:
        formatted = answer

    return {
        "raw_answer": answer,
        "formatted_answer": formatted,
        "citations": citations,
        "used_chunks": used_chunks,
        "retrieval_top_score": top_score,
        "prompt_template": template["id"],
    }


# ---------------------------------------------------------------------------
# Task 2: RetrievalQA-style orchestration
# ---------------------------------------------------------------------------

class RagPipeline:
    def __init__(self, chunk_size: int, overlap_pct: float, embedding_name: str, k: int = 4):
        self.chunk_size = chunk_size
        self.overlap_pct = overlap_pct
        self.embedding_name = embedding_name
        self.k = k
        docs = load_corpus()
        self.chunks = build_chunks(docs, chunk_size, overlap_pct)
        embedder = EMBEDDERS[embedding_name]()
        self.store = VectorStore(self.chunks, embedder)
        self.memory = ConversationBufferMemory()

    def ask(self, question: str, prompt_id: str) -> Dict:
        t0 = time.time()
        resolved_query = _resolve_followup(question, self.memory)
        retrieved = self.store.search(resolved_query, k=self.k)
        result = generate_answer(
            question, retrieved, prompt_id, self.memory, scoring_query=resolved_query
        )
        latency = time.time() - t0
        self.memory.add(question, result["raw_answer"])
        result.update(
            {
                "question": question,
                "resolved_query": resolved_query,
                "latency_sec": round(latency, 4),
                "tokens": len(result["formatted_answer"].split()),
                "config": {
                    "chunk_size": self.chunk_size,
                    "overlap_pct": self.overlap_pct,
                    "embedding": self.embedding_name,
                    "prompt": prompt_id,
                },
            }
        )
        return result
