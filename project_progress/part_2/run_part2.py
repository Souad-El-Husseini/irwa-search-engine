"""Run IRWA Part 2 from the repository root:

    python -m project_progress.part_2.run_part2

Writes ranking outputs, instructor-query metrics, and blank CSVs for human judges.
Does NOT fabricate the required manual relevance labels.
"""
from pathlib import Path
import argparse, json
import pandas as pd
from .prepare_data import prepare_data
from .search_engine import load_processed, build_inverted_index, calculate_idf, normalize_tf, rank_documents, preprocess, stopword_set
from .evaluation import evaluate_qrels, aggregate, evaluate

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'
RESULTS=ROOT/'results'
VALIDATION={1:'women full sleeve sweatshirt cotton',2:'men slim jeans blue'}
CUSTOM=[
    'Women pyjamas',
    'Blue navy jeans for women',
    'Men dotted tie',
    'Women silk dress',
    'Men brown pant',
]

def run(k=10, prepare_missing=True):
    RESULTS.mkdir(exist_ok=True,parents=True)
    if prepare_missing and not (DATA/'index_info.csv').exists():
        prepare_data(DATA)
    products, index_df=load_processed(DATA)
    idx, tf, df=build_inverted_index(index_df)
    idf=calculate_idf(df, len(index_df))
    norm=normalize_tf(tf)
    stops=stopword_set()
    products_by_pid=products.set_index('pid')
    validation_df=pd.read_csv(DATA/'validation_labels.csv')
    assert set(validation_df.columns)>={'pid','query_id','labels'}
    assert not validation_df.duplicated(['query_id','pid']).any()
    if not set(validation_df.labels.unique()).issubset({0,1}):
        raise ValueError('Validation labels must be 0/1')
    v_rows=[]; v_metrics=[]; judged_debug=[]; ranking_rows=[]; human_rows=[]; short_human_rows=[]; per_query_counts=[]

    for qid,query in VALIDATION.items():
        ranked=rank_documents(query,idx,norm,idf,stops)
        known=validation_df[validation_df.query_id==qid]
        qrels=dict(zip(known.pid,known.labels.astype(int)))
        m, judged=evaluate_qrels(query,ranked,qrels,k)
        v_metrics.append(m)
        v_rows.append({'Query':qid,'P@10':m['Precision'],'R@10':m['Recall'],
                      'AP@10':m['AP'],'F1@10':m['F1'],'NDCG@10':m['NDCG']})
        judged_rank={pid:r for r,(pid,_) in enumerate(judged,1)}
        for pid,label in qrels.items():
            judged_debug.append({'query_id':qid,'query':query,'pid':pid,'relevance':label,
                                 'retrieved':pid in {x for x,_ in judged},
                                 'judged_pool_rank':judged_rank.get(pid,None)})
        per_query_counts.append({'query_id':f'validation-{qid}','query':query,
                                 'matches':len(ranked),'judged_retrieved':len(judged),
                                 'known_relevant':sum(qrels.values())})
    agg=aggregate(v_metrics)
    vtable=pd.DataFrame(v_rows)
    vtable.to_csv(RESULTS/'validation_metrics.csv',index=False,float_format='%.3f')
    pd.DataFrame([{'MAP@10':agg['MAP'],'MRR@10':agg['MRR']}]).to_csv(RESULTS/'validation_aggregate.csv',index=False,float_format='%.3f')
    pd.DataFrame(judged_debug).to_csv(RESULTS/'validation_qrels_debug.csv',index=False)

    for qid,query in enumerate(CUSTOM,1):
        ranked=rank_documents(query,idx,norm,idf,stops)
        per_query_counts.append({'query_id':f'custom-{qid}','query':query,
                                 'matches':len(ranked),'judged_retrieved':None,'known_relevant':None})
        for rank,(pid,score) in enumerate(ranked,1):
            prod=products_by_pid.loc[pid]
            product_info={'query_id':qid,'query':query,'rank':rank,'pid':pid,
                          'tfidf_score':score,'title':prod.get('title',''),
                          'description':prod.get('description',''),
                          'brand':prod.get('brand',''),
                          'sub_category':prod.get('sub_category',''),
                          'product_details':prod.get('product_details','')}
            # Entire result set needs judges; no automatic relevance labels.
            human_rows.append({**product_info,'label':''})
            if rank<=k:short_human_rows.append({**product_info,'label':''})
            if rank<=20:ranking_rows.append(product_info)
    pd.DataFrame(ranking_rows).to_csv(RESULTS/'custom_query_top20_ranked.csv',index=False,float_format='%.6f')
    pd.DataFrame(per_query_counts).to_csv(RESULTS/'query_counts.csv',index=False)
    template=DATA/'human_judgments_ALL_BLANK.csv'
    if not template.exists():
        pd.DataFrame(human_rows).to_csv(template,index=False,float_format='%.6f')
    top_template=DATA/'human_judgments_TOP10_BLANK.csv'
    if not top_template.exists():
        pd.DataFrame(short_human_rows).to_csv(top_template,index=False,float_format='%.6f')
    metadata={'num_products':len(index_df),'vocabulary_size':len(idx),
              'fields':list(index_df.columns[1:]),'k':k,'tf':'raw term counts normalized by L2',
              'idf':'ln(N/df)','ranking':'TF-IDF weighted query-document dot product (not cosine)',
              'retrieval':'AND intersect across seven fields',
              'validation_protocol':'condensed to judged retrieved products; unknown unjudged ignored',
              'ap_denominator':'min(K, total relevant in judged pool)',
              'validation_judgments':int(len(validation_df)),
              'custom_queries':CUSTOM}
    (RESULTS/'methodology.json').write_text(json.dumps(metadata,indent=2))
    print('\nINSTRUCTOR VALIDATION (K=%d)'%k)
    print(vtable.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('MAP@%d %.3f  MRR@%d %.3f'%(k,agg['MAP'],k,agg['MRR']))
    print('\nMATCH COUNTS')
    print(pd.DataFrame(per_query_counts).to_string(index=False))
    print('\nYour group must fill the blank label column in:')
    print(' ',template)
    print(' ',top_template)
    print('Then run: python -m project_progress.part_2.evaluate_human --file data/human_judgments_ALL_BLANK.csv')
    return metadata, vtable, agg

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--k',type=int,default=10)
    p.add_argument('--skip-prepare',action='store_true')
    args=p.parse_args()
    run(args.k,not args.skip_prepare)
