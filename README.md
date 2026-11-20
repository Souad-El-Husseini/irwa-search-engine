# IRWA Search Engine — 2026

**Course:** Information Retrieval and Web Analytics (IRWA)  
**Project:** Fashion product search engine  
**Status:** Parts 1 and 2 completed; Part 3 ranking implementation and results uploaded (final report pending). Part 4 is a future project stage.

This repository contains Python code for a local information retrieval engine over the provided fashion-products dataset. Part 1 performs text preprocessing. Part 2 builds a seven-field inverted index, retrieves documents using Boolean **AND**, ranks matches using a TF-IDF weighted dot product, and evaluates the results with instructor-supplied and custom relevance judgments.

## Repository layout

```text
irwa-search-engine/
├── data/
│   ├── README.md
│   ├── human_judgments_ALL_BLANK.csv          # Completed 0/1 judgments; legacy filename
│   ├── human_judgments_TOP10_BLANK.csv
│   ├── processed_data.zip
│   ├── validation_labels.csv                  # Instructor relevance judgments
│   └── fashion_products_dataset.json         # Obtain from course; keep locally
├── project_progress/
│   ├── part_1/
│   │   ├── IRWA-2026-u231910-u279092-279112-part-1.pdf
│   │   └── preprocessing.ipynb             # Team's original Part 1 notebook
│   ├── part_2/
│   │   ├── IRWA-2026-u231910-u279092-u279112-part-2.pdf # Part 2 report 
│   │   ├── README.md                         # Detailed Part 2 methodology and results 
│   │   ├── evaluate_human.py                 # Custom-query evaluation 
│   │   ├── evaluation.py                     # Retrieval metrics 
│   │   ├── part2-indexing-and-evaluation-complete.ipynb  # Complete Part 2 notebook 
│   │   ├── prepare_data.py                   # JSON to processed CSVs 
│   │   ├── run_part2.py                      # Corpus indexing and validation
│   │   └── search_engine.py                  # Indexing, AND retrieval, TF-IDF ranking
│   └── part_3/                               # Four ranking methods and experiments
│       ├── part3-ranking-and-filtering.ipynb
│       ├── part3_ranking.py
│       ├── run_part3.py
│       └── README.md
├── results/                                  # Reproducible CSV/JSON outputs
│   ├── part2/                                # Part 2 indexing and evaluation
│   │   ├── custom_query_top20_ranked.csv
│   │   ├── human_aggregate.csv
│   │   ├── human_metrics.csv
│   │   ├── methodology.json
│   │   ├── query_counts.csv
│   │   ├── validation_aggregate.csv
│   │   ├── validation_metrics.csv
│   │   └── validation_qrels_debug.csv
│   └── part3/                                # Part 3 ranking experiments
│       ├── README.md
│       ├── methodology.json
│       ├── part3_top20_all_methods.csv
│       ├── query_counts.csv
│       ├── smart_shopper_adjusted_top20_five_queries.csv
│       └── word2vec_top20_five_queries.csv
├── static/
│   └── styles/
│   │   ├── bootstrap.min.css
│   │   ├── bootstrap.min.css.map
│   │   └── custom.css
│   ├── image.png
│   └── logo.png
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── doc_details.html
│   ├── index.html
│   ├── results.html
│   └── stats.html
├── tests/
│   ├── test_part2.py                       # Basic retrieval/metric tests
│   └── test_part3.py
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt
└── web_app.py
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

For notebook execution, open `project_progress/part_2/part2-indexing-and-evaluation-complete.ipynb`, select the project's Python environment as the Jupyter kernel, and run cells in order.

## Outputs

| Output | Description |
|---|---|
| `results/part2/validation_metrics.csv` | Per-query scores for the two instructor validation queries |
| `results/part2/validation_aggregate.csv` | MAP@10 and MRR@10 for instructor queries |
| `results/part2/custom_query_top20_ranked.csv` | Ranked top-20 product listings for each custom query |
| `results/part2/query_counts.csv` | AND-retrieval match counts |
| `results/part2/human_metrics.csv` | Five custom-query metrics using completed relevance labels |
| `results/part2/human_aggregate.csv` | MAP@10 and MRR@10 for custom queries |
| `results/part2/methodology.json` | Evaluation settings, index fields, and model description |

**Evaluation method:** K = 10; the 40 instructor judgments cover only part of the collection, so the report uses a judged-only condensed ranking for those queries. Custom-query recall is based on labeled *retrieved* products, not all relevant products in the corpus. See [Part 2 README] for definitions, numerical results and limitations.

## Authorship and academic integrity

The team's original Part 1 and initial Part 2 notebooks are preserved. ChatGPT was used to assist with checking labels, improving documentation, and wording the report. The team is responsible for understanding the algorithms, getting ideas, writing codes, verifying decisions, and making accurate disclosures in the submitted PDF.
