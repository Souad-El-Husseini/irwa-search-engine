# Data for IRWA Parts 1 and 2

Keep existing `validation_labels.csv` and `human_judgments_ALL_BLANK.csv` as they are.
The archive `processed_data.zip` contains `fashion_products.csv` and
`index_info.csv`, required by both notebooks and by Part 2.

The notebooks automatically extract these CSVs into `data/` on first run.
To use command-line scripts, from the project root, run:

```bash
python -m zipfile -e data/processed_data.zip data
python -m project_progress.part_2.run_part2 --skip-prepare
python -m project_progress.part_2.evaluate_human --file data/human_judgments_ALL_BLANK.csv --k 10
```

The raw CSVs are ignored by `.gitignore`, so the compressed archive is the
portable dataset for this project. Do not replace the existing completed
judgments with a blank template.
