from pathlib import Path
import argparse
import pandas as pd
from .evaluation import evaluate, aggregate
from .run_part2 import ROOT, RESULTS


def evaluate_file(path: Path, k=10):
    judges=pd.read_csv(path)
    needed={'query_id','query','rank','pid','label'}
    if not needed.issubset(judges.columns):
        raise ValueError('Missing columns: '+str(needed-set(judges.columns)))
    if judges.label.isna().any():
        raise ValueError(f'{int(judges.label.isna().sum())} labels still blank. Review every row and enter 0 or 1.')
    if not set(judges.label.unique()).issubset({0,1}):
        raise ValueError('Every label must be exactly 0 or 1.')
    m=[]
    for qid,rows in judges.groupby('query_id',sort=True):
        rows=rows.sort_values('rank')
        if rows['pid'].duplicated().any():
            raise ValueError(f'Duplicate PID in query {qid}')
        labels=rows.label.astype(int).tolist()
        metrics=evaluate(labels,int(sum(labels)),k)
        m.append({'query_id':qid,'query':rows.iloc[0]['query'], **metrics})
    agg=aggregate(m)
    RESULTS.mkdir(exist_ok=True,parents=True)
    df=pd.DataFrame(m)
    df.to_csv(RESULTS/'human_metrics.csv',index=False,float_format='%.3f')
    pd.DataFrame([agg]).to_csv(RESULTS/'human_aggregate.csv',index=False,float_format='%.3f')
    print(df.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('MAP@%d %.3f; MRR@%d %.3f'%(k,agg['MAP'],k,agg['MRR']))
    print('LIMITATION: recall uses only reviewed retrieved documents, not all relevant corpus products.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--file',default='data/human_judgments_ALL_BLANK.csv')
    p.add_argument('--k',type=int,default=10)
    args=p.parse_args()
    evaluate_file(ROOT/args.file,args.k)
