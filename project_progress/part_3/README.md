# IRWA Part 3 — Ranking & Filtering (GitHub + Google Colab)

This folder extends the same Part 2 field-aware positional index and **Boolean AND** retrieval to compare TF-IDF cosine, BM25, a custom fashion-shopping score, and Word2Vec cosine ranking. No Part 2 file is replaced.

## Adding the files to the repository

Upload **all four files in this folder** to `project_progress/part_3/` in the existing `irwa-search-engine` GitHub repository. The notebook needs the two `.py` files to work. Keep the already tracked `data/processed_data.zip` in `data/`, containing `fashion_products.csv` and `index_info.csv`.

## Running in the cloud (no local Python installation)

1. Commit the Part 3 folder to GitHub on the `main` branch.
2. Open [the Part 3 notebook in Google Colab](https://colab.research.google.com/github/Souad-El-Husseini/irwa-search-engine/blob/main/project_progress/part_3/part3-ranking-and-filtering.ipynb).
3. Choose **Runtime → Run all**. Colab clones this repository into its cloud workspace, installs missing dependencies, and imports the existing Part 2 code.
4. Review the results for the five original queries. The final cell downloads a ZIP with outputs generated under `results/part3/`.
5. Extract the ZIP and upload the useful CSV/JSON outputs to the GitHub `results/part3/` directory. The `.model` file is optional because it can be regenerated.

**Important:** GitHub stores and displays `.ipynb` files but does not execute their cells. Running in Colab does not automatically commit generated CSVs or the notebook's outputs back to GitHub.

## Code inputs

- From GitHub: `project_progress/part_2/search_engine.py` and related modules.
- From GitHub: `data/processed_data.zip` containing the two processed Part 2 CSVs. The ranker unpacks it when the raw CSVs are absent.
- Five queries from the Part 2 report, preserving strict conjunctive AND semantics.

## Results

The notebook outputs:
- `results/part3/part3_top20_all_methods.csv`
- `results/part3/word2vec_top20_five_queries.csv`
- `results/part3/query_counts.csv`
- `results/part3/methodology.json`
- `results/part3/word2vec.model` (optional to store on GitHub)

Rankings should never include products failing all query terms. A query that returns fewer than 20 matching products has a shorter ranking list.

The Part 3 report draft is separate and needs actual experimental ranking tables before submission.
