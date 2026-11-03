"""Generate the two data/ CSV files needed by Part 2 from the course JSON.

Follows the initial (preprocessing) section of the supplied Part 1 notebook;
it does not rerun its separate optional EDA, charts, or spaCy NER.
"""
from pathlib import Path
import pandas as pd
from .search_engine import preprocess, stopword_set, FIELDS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'


def details_to_text(x):
    return ' '.join(str(v) for d in x if isinstance(d, dict) for v in d.values()) if isinstance(x, list) else ''


def prepare_data(data_dir: Path = DATA) -> tuple[Path,Path]:
    dataset = data_dir / 'fashion_products_dataset.json'
    if not dataset.exists():
        raise FileNotFoundError(f'Put the course dataset at {dataset}')
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(dataset)
    if not df['pid'].is_unique or df.pid.isna().any():
        raise ValueError('Product IDs must be nonmissing and unique')
    for field in ('actual_price','selling_price'):
        df[field] = pd.to_numeric(df[field].astype('string').str.replace(',','',regex=False).str.strip(),errors='raise')
    df['discount'] = pd.to_numeric(df['discount'].astype('string').str.extract(r'^\s*(\d+(?:\.\d+)?)\s*%\s*off\s*$',expand=False),errors='coerce') / 100
    df['average_rating'] = pd.to_numeric(df['average_rating'].astype('string').str.strip(),errors='coerce')
    stops = stopword_set()
    idx = pd.DataFrame({'pid':df.pid})
    for col in FIELDS:
        if col == 'product_details':
            idx[col] = df[col].apply(details_to_text).apply(lambda x: preprocess(x,stops))
        else:
            idx[col] = df[col].apply(lambda x: preprocess(x,stops))
    a, b = data_dir/'fashion_products.csv', data_dir/'index_info.csv'
    df.to_csv(a,index=False)
    idx.to_csv(b,index=False)
    print(f'Saved {len(df):,} products to {a} and {b}')
    return a,b


if __name__=='__main__':
    prepare_data()
