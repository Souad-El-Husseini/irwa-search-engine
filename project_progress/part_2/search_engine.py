
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

FIELDS = ("title", "description", "brand", "category", "sub_category", "seller", "product_details")
_STEMMER = PorterStemmer()
_TOKENIZER = RegexpTokenizer(r"\w+")


def stopword_set() -> set[str]:
    try:
        words = set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError("NLTK English stopwords missing. Run: python -m nltk.downloader stopwords") from exc
    words.difference_update({"no", "not"})
    return words


def preprocess(text: Any, stops: set[str] | None = None) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    if stops is None:
        stops = stopword_set()
    return [_STEMMER.stem(t) for t in _TOKENIZER.tokenize(text.lower()) if t not in stops]


def parse_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            out = ast.literal_eval(value)
            if isinstance(out, list) and all(isinstance(t, str) for t in out):
                return out
        except (ValueError, SyntaxError):
            pass
    return []


def load_processed(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = data_dir / "fashion_products.csv", data_dir / "index_info.csv"
    if not a.exists() or not b.exists():
        raise FileNotFoundError("Part 1 CSVs missing. Run: python -m project_progress.part_2.prepare_data")
    products = pd.read_csv(a, low_memory=False)
    tokens = pd.read_csv(b, low_memory=False)
    if not products.pid.is_unique or not tokens.pid.is_unique:
        raise ValueError("Duplicate product identifiers in Part 1 output")
    if set(products.pid) != set(tokens.pid):
        raise ValueError("Part 1 product IDs differ between files")
    if any(field not in tokens.columns for field in FIELDS):
        raise ValueError("Part 1 index_info.csv must contain the seven searchable fields")
    return products, tokens


def build_inverted_index(index_df: pd.DataFrame):
    idx = defaultdict(lambda: defaultdict(dict))
    tf = defaultdict(Counter)
    for item in index_df.to_dict("records"):
        pid = str(item["pid"])
        for field in FIELDS:
            positions_for_term = defaultdict(list)
            for pos, term in enumerate(parse_tokens(item.get(field))):
                positions_for_term[term].append(pos)
            for term, positions in positions_for_term.items():
                idx[term][field][pid] = positions
                tf[pid][term] += len(positions)
    df = {term: len({pid for field in field_dict.values() for pid in field})
          for term, field_dict in idx.items()}
    return idx, dict(tf), df


def conjunctive_search(query: str, idx, stops: set[str] | None = None) -> list[str]:
    terms = list(dict.fromkeys(preprocess(query, stops)))
    if not terms:
        return []
    postings = []
    for term in terms:
        if term not in idx:
            return []
        candidates = set().union(*(set(v.keys()) for v in idx[term].values()))
        postings.append(candidates)
    return sorted(set.intersection(*postings))


def calculate_idf(df: dict[str,int], document_count: int) -> dict[str,float]:
    return {t: math.log(document_count / freq) for t, freq in df.items() if freq > 0}


def normalize_tf(tf: dict[str,Counter]) -> dict[str,dict[str,float]]:
    out = {}
    for pid, counts in tf.items():
        norm = math.sqrt(sum(v*v for v in counts.values()))
        out[pid] = {term: value/norm for term, value in counts.items()} if norm else {}
    return out


def rank_documents(query: str, idx, normalized_tf, idf,
                   stops: set[str] | None = None) -> list[tuple[str,float]]:
    qterms = preprocess(query, stops)
    candidates = conjunctive_search(query, idx, stops)
    if not qterms or not candidates:
        return []
    qcounts = Counter(qterms)
    qnorm = math.sqrt(sum(x*x for x in qcounts.values()))
    qweights = {t: (n/qnorm) * idf.get(t, 0.0) for t,n in qcounts.items()}
    ranked = []
    for pid in candidates:
        score = sum(normalized_tf.get(pid, {}).get(t, 0.0) * idf.get(t, 0.0) * qw
                    for t, qw in qweights.items())
        ranked.append((pid, score))
    return sorted(ranked, key=lambda p: (-p[1], p[0]))
