# Part 2 — Indexing and Evaluation

## Objective and inputs

Part 2 turns the prepared fashion catalog into a ranked search system and evaluates retrieval quality. The original Part 1 work is retained in `../part_1/preprocessing.ipynb`. The instructor's dataset has **28,080** unique product IDs (`pid`); the build produces **22,670** distinct indexed terms.

### Implementation files

| File | Role |
|---|---|
| `prepare_data.py` | Reproduces core Part 1 preprocessing from the source JSON; outputs `fashion_products.csv` and `index_info.csv` |
| `search_engine.py` | Term processing, positional index construction, AND retrieval, TF/DF/IDF calculation, TF-IDF scoring |
| `evaluation.py` | Precision@K, Recall@K, AP@K, F1@K, RR@K, MAP, MRR, DCG@K, NDCG@K |
| `run_part2.py` | Runs queries, saves validation scores, custom rankings and blank templates where missing |
| `evaluate_human.py` | Evaluates completed 0/1 judgments in a CSV, saves per-query and aggregate results |
| `indexing_and_evaluation.ipynb` | Guided notebook referencing the same Python modules |
| `part2-indexing-original.ipynb` | Preserved initial team indexing notebook |

## Indexing and retrieval design

1. Index seven text fields: `title`, `description`, `brand`, `category`, `sub_category`, `seller`, `product_details`. Numeric prices, ratings and stock remain metadata, not search terms.
2. Apply lowercase conversion, NLTK `RegexpTokenizer(r"\w+")`, English stopword removal (retaining `no` and `not`), then Porter stemming. Apply the same normalization to queries.
3. Build a **positional inverted index** mapping each normalized term to its field and matching `pid` with word positions. Use unique PIDs to calculate document frequency, while repeated occurrences contribute to term frequency.
4. Use **Boolean AND** retrieval: a product is eligible only if it contains *every* processed query term, allowing terms to appear in different indexed fields.
5. Let `N` be collection size, `df(t)` the number of products containing term `t`, and `tf(t,d)` its raw count in product `d`. The IDF is `ln(N / df(t))`; document and query raw term-frequency vectors are separately L2-normalized, multiplied by IDF and then combined by a **weighted dot product**. This is *not* a full cosine score because the final TF-IDF vectors are not L2-normalized. Scores are sorted descending with `pid` as tie-breaker. No field-specific weight is used.

The field `product_details` is converted to searchable **attribute values**, not attribute names. The extra term-position information permits possible later phrase/proximity work but is not required for the Part 2 AND operation.

## Run commands

From the repository root (`irwa-search-engine`), after installing requirements and NLTK stopwords as described in the [root README](../../README.md):

```powershell
python -m project_progress.part_2.run_part2
python -m project_progress.part_2.evaluate_human --file data/human_judgments_ALL_BLANK.csv --k 10
python -m unittest discover -s tests -v
```

The completed file retains the filename `human_judgments_ALL_BLANK.csv` for compatibility with the original command. **Its label column is filled for all 327 retrieved query/product pairs.** If you change labels, save the CSV, rerun `evaluate_human`, and update the report's numbers. Running `run_part2` first does not overwrite this sheet if it already exists.

## Five custom queries

| ID | Query | AND matches | Relevant in judged retrieved set |
|---:|---|---:|---:|
| 1 | Women pyjamas | 193 | 76 |
| 2 | Blue navy jeans for women | 43 | 6 |
| 3 | Men dotted tie | 3 | 3 |
| 4 | Women silk dress | 32 | 5 |
| 5 | Men brown pant | 56 | 47 |

**Labels:** each retrieved product was assigned binary relevance `1` or `0` based on its apparent suitability for the query. The initial judgments were entered and checked with ChatGPT assistance. Your group must review ambiguous cases, including differences between the title and detailed product description, before claiming they are expert-reviewed judgments. Never alter relevance labels merely to raise the scores.

## Evaluation definitions and convention

Let `K = 10`, `R` be the count of judged relevant products for a given query, and `rel_i` be the binary relevance at rank `i`.

- **Precision@K** = relevant results in first K / K. If fewer than K results are retrieved, remaining positions count as non-relevant.
- **Recall@K** = relevant results in first K / R. For custom queries R includes only *judged retrieved* products, and this is not true full-corpus recall.
- **Average Precision@K** = sum of precision at each relevant rank up to K / `min(K, R)`. This is the convention used in the supplied implementation; specify it when comparing results with other systems.
- **F1@K** = `2 * Precision@K * Recall@K / (Precision@K + Recall@K)`, or zero when both inputs are zero.
- **MAP@K** = average AP@K across queries.
- **MRR@K** = average `1 / rank_of_first_relevant_result`, or zero if no relevant result appears in top K.
- **NDCG@K** = DCG@K / ideal DCG@K, with binary gains and logarithmic position discount; zero if ideal gain is zero.

**Instructor validation:** The file `validation_labels.csv` contains only 20 judged items for each of two predefined queries. Unjudged items must not automatically be considered irrelevant. The evaluation ranks all products, **condenses the ranked list to judged retrieved items**, and keeps known relevant judged items that AND failed to retrieve in the Recall/AP/ideal-DCG denominators. Thus the reported scores measure a *partial-judgment protocol*, not exhaustive full-corpus relevance. The instructor did not fix K, so K=10 is a documented choice.

### Instructor queries — numerical results (three decimals)

| Query | P@10 | R@10 | AP@10 | F1@10 | NDCG@10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.769 | 1.000 | 0.870 | 1.000 |
| 2 | 0.600 | 0.600 | 0.549 | 0.600 | 0.706 |

| MAP@10 | MRR@10 |
|---:|---:|
| 0.775 | 1.000 |

Instructor query IDs: 1 = `women full sleeve sweatshirt cotton`; 2 = `men slim jeans blue`.

### Custom queries — completed sheet used in the student's run

| ID | P@10 | R@10 | AP@10 | F1@10 | RR@10 | NDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.132 | 1.000 | 0.233 | 1.000 | 1.000 |
| 2 | 0.600 | 1.000 | 0.931 | 0.750 | 1.000 | 0.978 |
| 3 | 0.300 | 1.000 | 1.000 | 0.462 | 1.000 | 1.000 |
| 4 | 0.500 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 |
| 5 | 1.000 | 0.213 | 1.000 | 0.351 | 1.000 | 1.000 |

| MAP@10 | MRR@10 |
|---:|---:|
| 0.986 | 1.000 |

These results reflect 47 (not 48) relevant products in the custom query `Men brown pant`. If the group uses a different labeling sheet, **rerun the metrics and update the table and PDF**; do not mix versions.

## Observations and limitations

- `Women pyjamas` has perfect top-10 precision but low conditional recall because 76 retrieved products were marked relevant: 10/76 = 0.132. Raising K may retrieve more of them but changes the evaluated cutoff.
- `Men dotted tie` has only three matched products; all are marked relevant, but Precision@10 is 3/10 = 0.300 because missing positions count as zero.
- `Blue navy jeans for women` has six relevant matches among the 43 retrieved products, all in top ten; `Women silk dress` has five judged relevant matches, all in top ten.
- Strict AND misses terms expressed through synonyms and can exclude relevant items; some long or noisy metadata may drive superficial keyword matches; field importance is not modeled, and prices, ratings, stock and semantics do not affect ranking.
- Instructor ground truth is sparse; custom recall considers only retrieved documents. A full recall estimate would require judging additional unreturned products.

Future improvements include field-aware weights, more robust attribute cleaning, query expansion, optional relaxed matching, BM25 or semantic ranking, and numeric/availability filters.

## Reproducibility and submission

Expected CSV files include `results/validation_metrics.csv`, `results/validation_aggregate.csv`, `results/custom_query_top20_ranked.csv`, `results/human_metrics.csv`, `results/human_aggregate.csv`, `results/query_counts.csv` and `results/methodology.json`. The evaluation functions can be tested without rerunning the full corpus: `python -m unittest discover -s tests -v`.

The full raw dataset is excluded via `.gitignore`; evaluators must place the instructor's copy at `data/fashion_products_dataset.json`. Publish the final PDF at `project_progress/part_2/IRWA_Part2_Report.pdf` and include the GitHub URL, student IDs and actual release tag in that PDF. The exact tag format must be checked against this year's instructions.

### AI assistance and responsibilities

ChatGPT assisted with code completion/debugging, testing, drafting these READMEs and the report, and checking/suggesting relevance labels. The uploaded labels include human entries and AI-assisted review; these are not independently verified expert labels unless all teammates complete that review. The group is responsible for reproducing, understanding and validating the submitted work, consistent with the course AI-use policy.
