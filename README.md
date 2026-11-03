# IRWA Search Engine — 2026

**Course:** Information Retrieval and Web Analytics (IRWA)  
**Project:** Fashion product search engine  
**Status:** Part 2 (Indexing and Evaluation) implemented and evaluated; Part 3 and Part 4 are future project stages.

This repository contains Python code for a local information retrieval engine over the provided fashion-products dataset. Part 1 performs text preprocessing. Part 2 builds a seven-field inverted index, retrieves documents using Boolean **AND**, ranks matches using a TF-IDF weighted dot product, and evaluates the results with instructor-supplied and custom relevance judgments.

## Repository layout

```text
irwa-search-engine/
├── data/
│   ├── validation_labels.csv                  # Instructor relevance judgments
│   ├── human_judgments_ALL_BLANK.csv          # Completed 0/1 judgments; legacy filename
│   └── fashion_products_dataset.json         # Obtain from course; keep locally
├── project_progress/
│   ├── part_1/preprocessing.ipynb             # Team's original Part 1 notebook
│   └── part_2/
│       ├── README.md                         # Detailed Part 2 methodology and results
│       ├── part2-indexing-original.ipynb     # Team's initial Part 2 notebook
│       ├── indexing_and_evaluation.ipynb     # Reproducible overview notebook
│       ├── prepare_data.py                   # JSON to processed CSVs
│       ├── search_engine.py                  # Indexing, AND retrieval, TF-IDF ranking
│       ├── evaluation.py                     # Retrieval metrics
│       ├── run_part2.py                      # Corpus indexing and validation
│       ├── evaluate_human.py                 # Custom-query evaluation
│       └── IRWA_Part2_Report.pdf             # Part 2 written report
├── results/                                  # Reproducible CSV outputs
├── tests/test_part2.py                       # Basic retrieval/metric tests
├── requirements.txt
└── README.md
```

The large original dataset and two generated intermediate CSVs (`fashion_products.csv`, `index_info.csv`) are **not tracked** under the existing `.gitignore`. Product identifiers remain unchanged so results can be joined against `validation_labels.csv` and the complete judgment sheet.

## Run Part 2 locally

**Prerequisites:** Python 3, the instructor-provided `fashion_products_dataset.json` placed at `data/fashion_products_dataset.json`, and a terminal opened in the `irwa-search-engine` root.

### Windows (VS Code PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m project_progress.part_2.run_part2
python -m project_progress.part_2.evaluate_human --file data/human_judgments_ALL_BLANK.csv --k 10
python -m unittest discover -s tests -v
```

The label file is **already filled in** (327 rows). Its original `ALL_BLANK` suffix is a legacy filename, not a description of its current contents. Do not replace it with the unlabeled template when updating the repository.

If PowerShell blocks environment activation, use `cmd` with `.venv\Scripts\activate.bat`, or run ` .venv\Scripts\python.exe -m ...` without activating. On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.

### Re-run without regenerating Part 1 preprocessing

If both `data/fashion_products.csv` and `data/index_info.csv` already exist, you may use:

```powershell
python -m project_progress.part_2.run_part2 --skip-prepare
```

For notebook execution, open `project_progress/part_2/indexing_and_evaluation.ipynb`, select the project's Python environment as the Jupyter kernel, and run cells in order.

## Outputs

| Output | Description |
|---|---|
| `results/validation_metrics.csv` | Per-query scores for the two instructor validation queries |
| `results/validation_aggregate.csv` | MAP@10 and MRR@10 for instructor queries |
| `results/custom_query_top20_ranked.csv` | Ranked top-20 product listings for each custom query |
| `results/query_counts.csv` | AND-retrieval match counts |
| `results/human_metrics.csv` | Five custom-query metrics using completed relevance labels |
| `results/human_aggregate.csv` | MAP@10 and MRR@10 for custom queries |
| `results/methodology.json` | Evaluation settings, index fields, and model description |

**Evaluation method:** K = 10; the 40 instructor judgments cover only part of the collection, so the report uses a judged-only condensed ranking for those queries. Custom-query recall is based on labeled *retrieved* products, not all relevant products in the corpus. See [Part 2 README] for definitions, numerical results and limitations.

## Authorship and academic integrity

The team's original Part 1 and initial Part 2 notebooks are preserved. ChatGPT was used to assist with checking labels, improving documentation, and wording the report. The team is responsible for understanding the algorithms, getting ideas, writing codes, verifying decisions, and making accurate disclosures in the submitted PDF.
