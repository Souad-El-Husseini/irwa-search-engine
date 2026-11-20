"""IRWA Part 3: rank the SAME AND candidates with four methods.

This file builds on project_progress.part_2.search_engine rather than replacing
Part 1 preprocessing or Part 2 indexing and retrieval.
"""
### Imports and dependencies
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re
import zipfile

import numpy as np
import pandas as pd

from project_progress.part_2.search_engine import (
    FIELDS, build_inverted_index, calculate_idf, conjunctive_search,
    load_processed, parse_tokens, preprocess, stopword_set,
)

### Predefined queries from part2
QUERIES = [
    "Women pyjamas",
    "Blue navy jeans for women",
    "Men dotted tie",
    "Women silk dress",
    "Men brown pant",
]

### Load the processed data (logic depending on where and how the dataset is located)
def _find_csv_files_in_zip(data_dir: Path) -> bool:
    """Use existing project data/processed_data.zip if it contains the two CSVs."""
    archive = data_dir / "processed_data.zip"
    wanted = ("fashion_products.csv", "index_info.csv")
    if not archive.exists():
        return False
    with zipfile.ZipFile(archive) as zf:
        paths = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
        if not all(name in paths for name in wanted):
            return False
        for filename in wanted:
            dest = data_dir / filename
            if not dest.exists():
                # Write only the expected files, not arbitrary archive paths.
                with zf.open(paths[filename]) as source, dest.open("wb") as out:
                    import shutil
                    shutil.copyfileobj(source, out)
    return True


def ensure_part2_data(data_dir: Path) -> None:
    """Use existing CSVs; otherwise restore them or regenerate from course JSON."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / name).exists() for name in ("fashion_products.csv", "index_info.csv")):
        return
    if _find_csv_files_in_zip(data_dir):
        return
    if (data_dir / "fashion_products_dataset.json").exists():
        from project_progress.part_2.prepare_data import prepare_data
        prepare_data(data_dir)
        return
    raise FileNotFoundError(
        "Missing Part 2 data. Provide data/fashion_products.csv and data/index_info.csv, "
        "data/processed_data.zip containing both, or the original "
        "data/fashion_products_dataset.json from the course."
    )

### Normalization functions for product attributes
def _number(value):
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    if isinstance(value, (float, int, np.number)):
        return float(value)
    s = str(value).strip().replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(match.group()) if match else None


def _rating_norm(row: dict) -> float:
    value = _number(row.get("average_rating_num", row.get("average_rating")))
    return 0.5 if value is None else min(max(value / 5.0, 0.0), 1.0)


def _discount_norm(row: dict) -> float:
    value = _number(row.get("discount_pct", row.get("discount")))
    if value is None:
        return 0.0
    # The original Part 1 data use percentages; Part 2 prepare_data.py
    # stores discounts as fractions in [0,1]. Both are supported.
    if "discount_pct" in row or "%" in str(row.get("discount", "")) or value > 1.0:
        value = value / 100.0
    return min(max(value, 0.0), 1.0)


def _stock_norm(row: dict) -> float:
    value = row.get("out_of_stock_bool", row.get("out_of_stock"))
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.5  # unknown status gets neutral credit
    if isinstance(value, (bool, np.bool_)):
        return 0.0 if value else 1.0
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "out of stock"}:
        return 0.0
    if s in {"false", "0", "no", "in stock"}:
        return 1.0
    return 0.5

### Ranker class, main part of Part 3
### Functionality: Load and organize data, 
#                  build inverted index, 
#                  process queries, 
#                  retrieve candidates, 
#                  calculate ranking scores, 
#                  return ranked products
class Part3Ranker:
    """Four scoring methods over a single shared Part 2 AND candidate set."""

    def __init__(self, products: pd.DataFrame, tokens: pd.DataFrame, stops=None):
        self.stops = stopword_set() if stops is None else stops
        self.products = products.copy()
        self.tokens = tokens.copy()
        self.products["pid"] = self.products.pid.astype(str)
        self.tokens["pid"] = self.tokens.pid.astype(str)
        self.products_by_pid = self.products.set_index("pid").to_dict("index")
        self.token_fields = {
            str(row["pid"]): {field: parse_tokens(row.get(field)) for field in FIELDS}
            for row in self.tokens.to_dict("records")
        }
        self.idx, self.raw_tf, self.df = build_inverted_index(self.tokens)
        self.N = len(self.token_fields)
        if self.N == 0:
            raise ValueError("No fashion products found")
        if set(self.token_fields) != set(self.products_by_pid):
            raise ValueError("Product identifiers differ between original and preprocessed data")
        self.idf = calculate_idf(self.df, self.N)  # original Part 2 IDF: ln(N/df)
        self.length = {pid: sum(counts.values()) for pid, counts in self.raw_tf.items()}
        self.avgdl = sum(self.length.values()) / self.N if self.N else 0.0
        self.doc_tfidf_norm = {
            pid: math.sqrt(sum((tf * self.idf.get(term, 0.0)) ** 2
                               for term, tf in counts.items()))
            for pid, counts in self.raw_tf.items()
        }
        self.w2v_model = None
        self.w2v_vectors = None
        self.w2v_pid_positions = None

    @classmethod
    def from_repo(cls, repo_root: Path):
        data_dir = Path(repo_root) / "data"
        ensure_part2_data(data_dir)
        products, tokens = load_processed(data_dir)
        return cls(products, tokens)

    def terms(self, query: str) -> list[str]:
        return preprocess(query, self.stops)

    def candidates(self, query: str) -> list[str]:
        return conjunctive_search(query, self.idx, self.stops)

    ### Ranking methods
    # tfidf + cosine
    def _cosine_for_candidates(self, query: str, candidates: list[str]) -> dict[str, float]:
        query_tf = Counter(self.terms(query))
        query_weights = {t: f * self.idf.get(t, 0.0) for t, f in query_tf.items()}
        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
        result = {}
        for pid in candidates:
            denominator = self.doc_tfidf_norm.get(pid, 0.0) * query_norm
            numerator = sum(
                self.raw_tf[pid].get(t, 0) * self.idf.get(t, 0.0) * q_weight
                for t, q_weight in query_weights.items()
            )
            result[pid] = numerator / denominator if denominator else 0.0
        return result

    def tfidf_cosine(self, query: str) -> list[tuple[str, float]]:
        candidates = self.candidates(query)
        return self._sorted(self._cosine_for_candidates(query, candidates))

    #BM25
    def bm25(self, query: str, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1>0 and 0<=b<=1")
        scores = {}
        for pid in self.candidates(query):
            score = 0.0
            doc_len = self.length.get(pid, 0)
            for term in set(self.terms(query)):
                freq = self.raw_tf[pid].get(term, 0)
                df = self.df[term]
                idf_bm25 = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * doc_len / self.avgdl)
                score += idf_bm25 * freq * (k1 + 1) / denom if denom else 0.0
            scores[pid] = score
        return self._sorted(scores)

    #custom scoring
    def custom(self, query: str):
        """0.60 cosine + 0.15 title match + 0.10 rating + 0.05 discount + 0.10 stock.

        These are DESIGN choices, not empirically optimized weights.
        Keep text relevance dominant. Values are normalized to [0,1].
        """
        candidates = self.candidates(query)
        cosine = self._cosine_for_candidates(query, candidates)
        terms = set(self.terms(query))
        scores = {}
        for pid in candidates:
            row = self.products_by_pid[pid]
            title_terms = set(self.token_fields[pid]["title"])
            title_coverage = len(terms & title_terms) / len(terms) if terms else 0.0
            scores[pid] = (
                0.60 * cosine[pid]
                + 0.15 * title_coverage
                + 0.10 * _rating_norm(row)
                + 0.05 * _discount_norm(row)
                + 0.10 * _stock_norm(row)
            )
        return self._sorted(scores)

    def train_word2vec(self, vector_size=100, window=5, epochs=10, seed=42):
        """Learn vectors on our own corpus, averaging all seven fields per product."""
        from gensim.models import Word2Vec
        sentences = [
            [term for field in FIELDS for term in self.token_fields[pid][field]]
            for pid in self.token_fields
        ]
        sentences = [s for s in sentences if s]
        self.w2v_model = Word2Vec(
            sentences=sentences, vector_size=vector_size, window=window,
            min_count=1, workers=1, sg=1, seed=seed, epochs=epochs,
        )
        pids = list(self.token_fields)
        vectors = [self._mean_vector([t for field in FIELDS
                                      for t in self.token_fields[pid][field]]) for pid in pids]
        self.w2v_vectors = np.asarray(vectors, dtype=np.float32)
        self.w2v_pid_positions = {pid: i for i, pid in enumerate(pids)}
        return self.w2v_model

    def _mean_vector(self, terms: list[str]):
        if self.w2v_model is None:
            raise RuntimeError("Train Word2Vec first")
        vectors = [self.w2v_model.wv[t] for t in terms if t in self.w2v_model.wv]
        if not vectors:
            return np.zeros(self.w2v_model.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0)

    def word2vec_cosine(self, query: str):
        if self.w2v_vectors is None:
            raise RuntimeError("Call train_word2vec() before ranking")
        candidates = self.candidates(query)
        if not candidates:
            return []
        query_vector = self._mean_vector(self.terms(query))
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return self._sorted({pid: 0.0 for pid in candidates})
        positions = [self.w2v_pid_positions[pid] for pid in candidates]
        matrix = self.w2v_vectors[positions]
        norms = np.linalg.norm(matrix, axis=1) * query_norm
        dot = matrix @ query_vector
        similarities = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0)
        return self._sorted(dict(zip(candidates, map(float, similarities))))

    @staticmethod
    def _sorted(scores: dict[str, float]) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda row: (-row[1], row[0]))

    def results_table(self, query: str, ranking: str, top_k=20):
        methods = {
            "TF-IDF cosine": self.tfidf_cosine,
            "BM25": self.bm25,
            "Custom": self.custom,
            "Word2Vec cosine": self.word2vec_cosine,
        }
        if ranking not in methods:
            raise ValueError("Unknown ranking method: " + ranking)
        rows = []
        for rank, (pid, score) in enumerate(methods[ranking](query)[:top_k], 1):
            product = self.products_by_pid[pid]
            rows.append({
                "rank": rank, "pid": pid,
                "title": product.get("title", ""),
                "brand": product.get("brand", ""),
                "score": round(score, 6),
                "average_rating": product.get("average_rating", None),
                "discount": product.get("discount", None),
                "out_of_stock": product.get("out_of_stock", None),
            })
        return pd.DataFrame(rows, columns=[
            "rank", "pid", "title", "brand", "score", "average_rating",
            "discount", "out_of_stock"
        ])
