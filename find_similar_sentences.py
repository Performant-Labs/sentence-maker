from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sparse_normalize
from unidecode import unidecode


try:
    import spacy
except ImportError as exc:  # pragma: no cover - guardrail for missing dependency
    raise SystemExit(
        "spaCy is required. Install it inside the conda env (conda install spacy)."
    ) from exc


DEFAULT_INPUT = pathlib.Path("input/sentences.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify Spanish sentences with very similar wording and structure."
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help="Path to the newline separated sentences file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Combined similarity threshold (0-1) for reporting a pair.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=25,
        help="Number of ANN neighbors to fetch per sentence before rescoring.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Maximum number of similar pairs to print.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional JSON file to write the detected pairs.",
    )
    parser.add_argument(
        "--embedding-model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for semantic similarity.",
    )
    parser.add_argument(
        "--spacy-model",
        default="es_core_news_md",
        help="spaCy model used to obtain POS tags (download if missing).",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("LEXICAL", "STRUCTURAL", "SEMANTIC"),
        default=(0.4, 0.2, 0.4),
        help="Weights applied to lexical, structural, and semantic similarity.",
    )
    return parser.parse_args()


def read_sentences(path: pathlib.Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    sentences: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            normalized = line.strip()
            if normalized:
                sentences.append(normalized)
    if not sentences:
        raise ValueError(f"No sentences were found in {path}")
    return sentences


def normalize_text(text: str) -> str:
    normalized = unidecode(text).lower()
    normalized = normalized.replace("¡", " ").replace("!", " ")
    normalized = normalized.replace("¿", " ").replace("?", " ")
    cleaned = []
    for ch in normalized:
        cleaned.append(ch if ch.isalnum() or ch.isspace() else " ")
    compact = " ".join("".join(cleaned).split())
    return compact


def build_tfidf(
    documents: Sequence[str],
    **vectorizer_kwargs,
) -> sparse.csr_matrix | None:
    if not documents:
        return None
    if all(not doc for doc in documents):
        return None
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    matrix = vectorizer.fit_transform(documents)
    if matrix.shape[1] == 0:
        return None
    return sparse_normalize(matrix)


def extract_pos_documents(nlp, sentences: Sequence[str], ngram_max: int = 3) -> List[str]:
    docs = []
    total = len(sentences)
    processed = 0
    counter_print_points = {1}
    step = max(1, total // 10)
    counter_print_points.update(range(step, total + 1, step))
    counter_print_points.add(total)
    for doc in nlp.pipe(sentences, batch_size=64):
        processed += 1
        if processed in counter_print_points:
            print(f"Processing sentences: {processed}/{total}")
        tags = [token.pos_ for token in doc if not token.is_space]
        grams = []
        for n in range(1, ngram_max + 1):
            if len(tags) < n:
                break
            for i in range(len(tags) - n + 1):
                grams.append("_".join(tags[i : i + n]))
        docs.append(" ".join(grams))
    return docs


def semantic_embeddings(
    model_name: str, sentences: Sequence[str], batch_size: int = 64
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(sentences),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def sparse_cosine(matrix: sparse.csr_matrix | None, i: int, j: int) -> float:
    if matrix is None:
        return 0.0
    return float(matrix[i].multiply(matrix[j]).sum())


@dataclass
class SimilarPair:
    score: float
    lexical: float
    structural: float
    semantic: float
    left_idx: int
    right_idx: int


def score_candidates(
    candidates: Iterable[Tuple[int, int]],
    lexical_mats: Sequence[sparse.csr_matrix | None],
    structural_mat: sparse.csr_matrix | None,
    embeddings: np.ndarray,
    weights: Tuple[float, float, float],
) -> List[SimilarPair]:
    weight_lexical, weight_structural, weight_semantic = weights
    scored: List[SimilarPair] = []
    for i, j in candidates:
        lexical_scores = [
            sparse_cosine(mat, i, j) for mat in lexical_mats if mat is not None
        ]
        lexical = sum(lexical_scores) / len(lexical_scores) if lexical_scores else 0.0
        structural = sparse_cosine(structural_mat, i, j)
        semantic = float(np.dot(embeddings[i], embeddings[j]))
        combined = (
            weight_lexical * lexical
            + weight_structural * structural
            + weight_semantic * semantic
        )
        scored.append(
            SimilarPair(
                score=combined,
                lexical=lexical,
                structural=structural,
                semantic=semantic,
                left_idx=i,
                right_idx=j,
            )
        )
    return scored


def build_candidate_pairs(embeddings: np.ndarray, candidates: int) -> set[Tuple[int, int]]:
    if embeddings.size == 0:
        return set()
    num_sentences = embeddings.shape[0]
    pair_set: set[Tuple[int, int]] = set()

    if faiss is not None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        sims, neighbors = index.search(embeddings, candidates + 1)
        del sims  # not needed beyond neighbor extraction
        for i, neigh_row in enumerate(neighbors):
            for j in neigh_row:
                if j == -1 or j == i:
                    continue
                pair = (i, j) if i < j else (j, i)
                pair_set.add(pair)
        return pair_set

    similarity_matrix = embeddings @ embeddings.T
    np.fill_diagonal(similarity_matrix, -np.inf)
    effective_candidates = min(candidates, num_sentences - 1)
    for i in range(num_sentences):
        row = similarity_matrix[i]
        if effective_candidates == num_sentences - 1:
            neighbor_idx = np.argsort(row)[::-1]
        else:
            part = np.argpartition(row, -effective_candidates)[-effective_candidates:]
            neighbor_idx = part[np.argsort(row[part])[::-1]]
        for j in neighbor_idx:
            if j <= i:
                continue
            pair_set.add((i, j))
    return pair_set


def format_pair_output(pair: SimilarPair, sentences: Sequence[str]) -> str:
    left = sentences[pair.left_idx]
    right = sentences[pair.right_idx]
    return (
        f"[{pair.score:.3f}] #{pair.left_idx+1} ↔ #{pair.right_idx+1}\n"
        f"  LEX={pair.lexical:.3f} STR={pair.structural:.3f} SEM={pair.semantic:.3f}\n"
        f"  • {left}\n"
        f"  • {right}"
    )


def write_json(
    output_path: pathlib.Path, pairs: Sequence[SimilarPair], sentences: Sequence[str]
) -> None:
    payload = []
    for pair in pairs:
        payload.append(
            {
                "score": float(pair.score),
                "lexical": float(pair.lexical),
                "structural": float(pair.structural),
                "semantic": float(pair.semantic),
                "left_index": int(pair.left_idx + 1),
                "right_index": int(pair.right_idx + 1),
                "left_sentence": sentences[pair.left_idx],
                "right_sentence": sentences[pair.right_idx],
            }
        )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_config_summary(
    args: argparse.Namespace, sentence_count: int | None = None
) -> None:
    print("\nConfiguration summary:")
    print(f"  • Input file          : {args.input}")
    if sentence_count is None:
        print("  • Sentences           : (pending - reading input)")
    else:
        print(f"  • Sentences           : {sentence_count}")
    print(f"  • Similarity threshold: {args.threshold}")
    print(f"  • Candidates per sent.: {args.candidates}")
    print(f"  • Report top          : {args.top}")
    print(f"  • Weights (L/S/Se)    : {args.weights}")
    print(f"  • Embedding model     : {args.embedding_model}")
    print(f"  • spaCy model         : {args.spacy_model}")
    print(f"  • Output path         : {args.output if args.output else 'stdout only'}\n")


def log_stage(timings: list[tuple[str, float]], label: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    timings.append((label, elapsed))
    print(f"{label} completed in {elapsed:.2f}s")


def main() -> None:
    args = parse_args()
    timings: list[tuple[str, float]] = []
    overall_start = time.perf_counter()
    print("Starting sentence similarity detection...")
    print_config_summary(args, sentence_count=None)

    stage_start = time.perf_counter()
    sentences = read_sentences(args.input)
    normalized_sentences = [normalize_text(sentence) for sentence in sentences]
    log_stage(timings, "Initialization (load + normalize sentences)", stage_start)
    print(f"Loaded {len(sentences)} sentences from {args.input}")

    try:
        stage_start = time.perf_counter()
        nlp = spacy.load(args.spacy_model)
        log_stage(timings, "Initialization (load spaCy model)", stage_start)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{args.spacy_model}' is not installed. "
            "Install it with: python -m spacy download es_core_news_md"
        ) from exc

    print("Extracting lexical features...")
    stage_start = time.perf_counter()
    word_tfidf = build_tfidf(
        normalized_sentences,
        ngram_range=(1, 2),
        min_df=1,
    )
    char_tfidf = build_tfidf(
        normalized_sentences,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )
    log_stage(timings, "Lexical features", stage_start)

    print("Extracting structural features via spaCy...")
    stage_start = time.perf_counter()
    pos_documents = extract_pos_documents(nlp, sentences)
    pos_tfidf = build_tfidf(
        pos_documents,
        ngram_range=(1, 3),
        min_df=2,
    )
    log_stage(timings, "Structural features", stage_start)

    print(f"Encoding sentences with {args.embedding_model}...")
    stage_start = time.perf_counter()
    embeddings = semantic_embeddings(args.embedding_model, sentences)
    log_stage(timings, "Sentence embeddings", stage_start)

    if faiss is None:
        print("Building candidate pairs via brute-force similarity (faiss not installed)...")
    else:
        print("Building candidate pairs via FAISS...")
    stage_start = time.perf_counter()
    candidate_pairs = build_candidate_pairs(embeddings, args.candidates)
    if not candidate_pairs:
        raise SystemExit("No candidate pairs found. Try increasing --candidates.")
    log_stage(timings, "Candidate generation", stage_start)
    print(f"Rescoring {len(candidate_pairs)} candidate pairs...")

    stage_start = time.perf_counter()
    scored_pairs = score_candidates(
        candidate_pairs, (word_tfidf, char_tfidf), pos_tfidf, embeddings, args.weights
    )
    filtered_pairs = [pair for pair in scored_pairs if pair.score >= args.threshold]
    filtered_pairs.sort(key=lambda pair: pair.score, reverse=True)
    log_stage(timings, "Scoring + filtering", stage_start)

    limit = args.top if args.top > 0 else len(filtered_pairs)
    top_pairs = filtered_pairs[:limit]

    if not top_pairs:
        print("No sentence pairs exceeded the similarity threshold.")
        return

    print(f"Top {len(top_pairs)} pairs (threshold={args.threshold}):")
    for pair in top_pairs:
        print(format_pair_output(pair, sentences))

    if args.output:
        write_json(args.output, top_pairs, sentences)
        print(f"\nSaved {len(top_pairs)} pairs to {args.output}")

    total_elapsed = time.perf_counter() - overall_start
    print("\nProcess summary:")
    print(f"  • Sentences processed     : {len(sentences)}")
    print(f"  • Candidate pairs evaluated: {len(candidate_pairs)}")
    print(f"  • Pairs >= threshold      : {len(filtered_pairs)}")
    print(f"  • Pairs reported          : {len(top_pairs)}")

    print("\nTiming summary:")
    for label, seconds in timings:
        print(f"  • {label:<30} {seconds:>7.2f}s")
    print(f"  • Total runtime                     {total_elapsed:>7.2f}s")


if __name__ == "__main__":
    main()
