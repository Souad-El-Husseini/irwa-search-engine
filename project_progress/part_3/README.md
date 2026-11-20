# IRWA Part 3 — Ranking & Filtering

## Overview

Part 3 extends our existing search engine from Part 2 by implementing and comparing four ranking methods while preserving Boolean AND retrieval.

## Ranking Methods

- TF-IDF + Cosine Similarity
- BM25
- Custom Smart Shopper Score
- Experimental adjusted Smart Shopper ranking (wrong-type and attribute penalties)
- Word2Vec + Cosine Similarity

## Custom Score

Our Smart Shopper Score uses TF-IDF cosine relevance (40%), product-type match in title or subcategory (20%), average rating (15%), discount (15%), and stock availability (10%). All terms are normalized to [0, 1]. Relevance and product type account for 60% of the score; ratings, savings, and availability account for the other 40%. These are proposed design weights, not empirically optimized parameters.

## Experimental Adjustment

We retain the original Smart Shopper Score as a baseline. The separate experimental method named `Smart Shopper Adjusted` uses

`Adjusted = Original Smart Shopper Score × product_type_factor × attribute_factor`

A recognized wrong clothing type gets factor **0.35** (matched: **1.0**), based on title and subcategory. A requested color, material, or pattern missing from title and product details gets **0.70** (matched: **1.0**). A query with no recognized clothing type or attribute receives a neutral factor of **1.0**. These values are starting heuristics, not validated optimal settings. All methods still use the same Boolean AND candidates.

## Project Files

- `part3-ranking-and-filtering.ipynb` — Main notebook containing the ranking experiments and results.
- `part3_ranking.py` — Implementation of the four ranking methods.
- `run_part3.py` — Optional script for running the ranking pipeline.
- `README.md` — Project documentation.

## Dataset

We reuse the 28,080 fashion products and the preprocessed data from Part 1, together with the inverted index and conjunctive retrieval methods from Part 2.

## Running the Notebook

1. Open the Part 3 notebook in Google Colab.
2. Select **Runtime → Run all**.
3. Review the ranking results for the five queries from Part 2.
4. Export the generated results and upload them to the GitHub repository.

## Outputs

Results will be saved under `results/part3/`, including:

- `part3_top20_all_methods.csv`
- `word2vec_top20_five_queries.csv`
- `smart_shopper_adjusted_top20_five_queries.csv`
- `query_counts.csv`
- `methodology.json`

The Word2Vec model can also be saved for reproducibility.

All ranking methods use the same AND retrieval candidates. Queries with fewer than 20 matching products return fewer than 20 results.

The Part 3 report draft is separate and needs actual experimental ranking tables before submission.

## Verification

Run `python -m unittest discover -s tests -v` from the repository root after installing `requirements.txt`. Part 3 test cases use synthetic products; run the notebook in Colab to verify the complete fashion corpus and generate actual ranking results.
