"""Reproduce the four rankings and top-20 output for the Part 2 queries.

From the repository root run:
    python -m project_progress.part_3.run_part3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .part3_ranking import Part3Ranker, QUERIES

ROOT = Path(__file__).resolve().parents[2]


def run(repo_root: Path = ROOT, vector_size=100, epochs=10):
    model = Part3Ranker.from_repo(repo_root)
    output_dir = repo_root / "results" / "part3"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {model.N:,} products; {len(model.df):,} index terms")
    print("Training Word2Vec on Part 1 processed fashion products ...")
    model.train_word2vec(vector_size=vector_size, epochs=epochs)
    model.w2v_model.save(str(output_dir / "word2vec.model"))

    all_rows = []
    counts = []
    for query_id, query in enumerate(QUERIES, 1):
        count = len(model.candidates(query))
        counts.append({"query_id": query_id, "query": query, "and_matches": count})
        for method in ("TF-IDF cosine", "BM25", "Custom", "Smart Shopper Adjusted", "Word2Vec cosine"):
            frame = model.results_table(query, method, top_k=20)
            frame.insert(0, "ranking_method", method)
            frame.insert(0, "query", query)
            frame.insert(0, "query_id", query_id)
            all_rows.append(frame)
        print(f"Query {query_id}: {query} -> {count} AND matches")
    ranks = pd.concat(all_rows, ignore_index=True)
    ranks.to_csv(output_dir / "part3_top20_all_methods.csv", index=False)
    ranks[ranks.ranking_method == "Word2Vec cosine"].to_csv(
        output_dir / "word2vec_top20_five_queries.csv", index=False)
    ranks[ranks.ranking_method == "Smart Shopper Adjusted"].to_csv(
        output_dir / "smart_shopper_adjusted_top20_five_queries.csv", index=False)
    pd.DataFrame(counts).to_csv(output_dir / "query_counts.csv", index=False)
    metadata = {
        "corpus_size": model.N, "vocabulary_size": len(model.df),
        "retrieval": "Part 2 Boolean AND over seven indexed fields",
        "tfidf": "original ln(N/df) IDF, final cosine-normalized vectors",
        "bm25": {"k1": 1.5, "b": 0.75},
        "custom_score": "Smart Shopper Score",
        "custom_weights": {"tfidf_cosine": 0.40, "product_type_match": 0.20,
                           "rating": 0.15, "discount": 0.15, "stock": 0.10},
        "product_type_rule": "Requested clothing type in title or subcategory; 0.5 neutral if unrecognized",
        "experimental_adjustments": {
            "method": "Smart Shopper Adjusted",
            "product_type_multiplier_on_mismatch": 0.35,
            "attribute_multiplier_on_mismatch": 0.70,
            "type_fields": ["title", "sub_category"],
            "attribute_fields": ["title", "product_details"],
            "no_recognized_query_factor": 1.0,
        },
        "word2vec": {"training": "Fashion corpus, skip-gram", "dimensions": vector_size,
                     "window": 5, "epochs": epochs, "seed": 42, "min_count": 1},
        "queries": QUERIES,
        "notes": "All rankings use exactly the same AND candidate set. Top-20 means up to 20 matches.",
    }
    (output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved results under {output_dir}")
    return ranks, pd.DataFrame(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run(vector_size=args.vector_size, epochs=args.epochs)
