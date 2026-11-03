# IRWA Final Project 2026 — Part 2: Indexing and Evaluation

**Working report draft — not final until human judging and team review.**

**Team:** [Actual names and student IDs]

**GitHub repository:** [URL]  
**Git tag:** [Confirm exact Part 2 tag format]

## 1. Objective

Extend the Part 1 fashion product preprocessing into a working search engine with an inverted index, Boolean AND retrieval, TF-IDF ranking, and relevance evaluation. The system was tested on 28,080 provided fashion products.

## 2. Index and preprocessing decisions

The seven searchable fields are `title`, `description`, `brand`, `category`, `sub_category`, `seller`, and values of `product_details`. A product retains its original `pid` and non-text metadata. The tokenizer, NLTK English stop words (preserving `no` and `not`) and Porter stemmer follow the original Part 1 notebook. Each inverted-index posting keeps the field and within-field token position. Term frequency counts repetitions; document frequency counts unique products containing the term.

Queries use AND retrieval over normalized terms, intersecting the relevant product-ID sets. Matching terms may occur in different fields of a single product.

## 3. TF-IDF scoring

We apply the team Part 2 convention:

- `idf(t) = ln(N / df(t))` where `N` is the number of documents and `df` the number of documents containing `t`.
- Normalize each product term-count vector by its Euclidean (L2) norm.
- Normalize query term counts by their Euclidean norm.
- Weight each document and query term by IDF and take their dot product.
- Rank AND-matched products by descending score, then ascending PID to break ties.

This is a TF-IDF weighted dot product, **not** final cosine normalization of the TF-IDF vectors.

## 4. Five custom queries

| Query | Search string | AND matches |
|---|---|---:|
| 1 | Women pyjamas | 193 |
| 2 | Blue navy jeans for women | 43 |
| 3 | Men dotted tie | 3 |
| 4 | Women silk dress | 32 |
| 5 | Men brown pant | 56 |

The group refined its original test queries to reduce empty-result queries under the strict AND requirement.

## 5. Predefined instructor validation — numeric results only

K = 10

| Query | P@10 | R@10 | AP@10 | F1@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.769 | 1.000 | 0.870 | 1.000 |
| 2 | 0.600 | 0.600 | 0.549 | 0.600 | 0.706 |

| MAP@10 | MRR@10 |
|---:|---:|
| 0.775 | 1.000 |

## 6. Evaluation protocol (separate from the numeric-only section)

The instructor provides twenty relevance judgments per predefined query, not judgments for all 28,080 products. We rank the full corpus first and preserve the order of products that are in the judged pool; unjudged retrieved products are excluded from this **condensed judged-only** ranking rather than assumed irrelevant. Judged relevant products not retrieved by AND still enter the denominators for Recall, AP and ideal DCG. Missing result slots count as zero in Precision@K. We select K=10, since the instructions do not specify K, and use the denominator `min(K, number of known relevant products)` for AP@K. These choices must be verified against the professor's expected protocol.

## 7. Description of evaluation metrics

- **P@K:** proportion of top K results that are relevant.
- **R@K:** proportion of all known relevant documents retrieved in the top K.
- **AP@K:** precision at relevant ranks, averaged over `min(K, #relevant)`.
- **F1@K:** harmonic mean of precision and recall.
- **MAP:** mean AP across queries.
- **MRR:** mean reciprocal rank of the first relevant result.
- **NDCG@K:** discount relevance gains by rank and divide by the best possible DCG, using binary relevance.

## 8. Interpretation of the two instructor queries (team review required)

The scores indicate that judged relevant items tend to appear early for Query 1; performance for Query 2 is lower. Both queries yield an early first relevant result, which raises MRR, but this does not mean all known relevant products are retrieved. The incomplete judgments and the strict AND constraint limit the conclusions. **Team: inspect the ranked products and adjust this interpretation based on concrete examples.**

## 9. Human-judged evaluation of five custom queries — REQUIRED

The generated `data/human_judgments_ALL_BLANK.csv` contains **327 unlabeled query/product results**. Each student must independently review assigned products using titles, descriptions and detailed attributes, then enter `1` for relevant to the intended shopper, `0` otherwise. Do not fabricate missing labels. After labeling, run the documented `evaluate_human` command.

**Paste the actual five-query metrics table from `results/human_metrics.csv` here after human judging.**

**Add your group's own analysis comparing each metric across the five queries.**

For custom queries, judging only retrieved products means recall cannot measure full-corpus retrieval completeness. State this limitation explicitly.

## 10. Limitations and improvements

- **Strict AND** excludes synonymous and partial matches; consider query expansion or relaxed retrieval for future parts.
- **Field-insensitive weighting** counts terms in title, seller and description equivalently; evaluate field weighting or attribute-aware search.
- **Data noise and repetitive descriptions** can inflate TF-IDF; improve cleaning and deduplication.
- **No structured filters** for availability, rating, discount or price; keep those fields numeric and add controlled filtering in later parts.
- **No semantic retrieval** means conceptually equivalent phrasing may miss; explore BM25 and semantic representations in Part 3.
- **Incomplete relevance ground truth** limits metrics; broaden expert assessments where feasible.

## 11. Reproducibility

See `README.md` in this folder. From the repository root, install `requirements.txt`, download NLTK English stopwords, then run `python -m project_progress.part_2.run_part2`. Verify with `python -m unittest discover -s tests -v`. Detailed numeric tables are saved under `results/`.

## 12. Submission and disclosure — TO COMPLETE

[Insert repository URL, correct tag, member IDs, manual-judgment results, and final conclusions. Rename the PDF to match the required course filename format. Upload it to Aula Global.]

**Sample disclosure — revise to match true use:** “ChatGPT assisted with coding and testing the AND search and evaluation metrics, improving reproducibility documentation, and structuring an initial report draft. The team ran and checked the code, reviewed the analysis, and manually labeled the relevance judgments.”

**Course material used:** Part 1 and Part 2 assignment PDFs, IRWA lecture slides, supplied preprocessing/indexing notebooks, fashion product dataset, and instructor relevance labels.
