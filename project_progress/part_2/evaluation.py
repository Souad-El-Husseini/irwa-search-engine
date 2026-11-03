"""Binary retrieval metrics and incomplete-judgment evaluation protocol."""
from __future__ import annotations
from dataclasses import dataclass
import math
from collections import defaultdict
from typing import Iterable
import pandas as pd


def _is_binary(labels):
    if any(x not in (0, 1) for x in labels):
        raise ValueError('Only human-provided binary labels 0/1 are allowed')


def precision_at_k(labels, k):
    if k <= 0:
        raise ValueError('k must be positive')
    _is_binary(labels)
    return sum(labels[:k]) / k


def recall_at_k(labels, total_relevant, k):
    _is_binary(labels)
    return sum(labels[:k]) / total_relevant if total_relevant else 0.0


def ap_at_k(labels, total_relevant, k):
    """AP at K: sum of P@i at relevant positions / min(K, #known relevant)."""
    _is_binary(labels)
    denominator = min(k,total_relevant)
    if denominator <= 0: return 0.0
    hits = 0
    total = 0.0
    for rank, label in enumerate(labels[:k],1):
        hits += label
        if label: total += hits/rank
    return total/denominator


def f1_at_k(labels, total_relevant, k):
    p,r=precision_at_k(labels,k),recall_at_k(labels,total_relevant,k)
    return 2*p*r/(p+r) if p+r else 0.0


def rr_at_k(labels,k):
    _is_binary(labels)
    return next((1/rank for rank,label in enumerate(labels[:k],1) if label),0.0)


def dcg_at_k(labels,k):
    _is_binary(labels)
    return sum((2**label-1)/math.log2(rank+1) for rank,label in enumerate(labels[:k],1))


def ndcg_at_k(labels,total_relevant,k):
    ideal = dcg_at_k([1]*min(total_relevant,k),k)
    return dcg_at_k(labels,k)/ideal if ideal else 0.0


def evaluate(labels, total_relevant, k=10):
    return dict(Precision=precision_at_k(labels,k), Recall=recall_at_k(labels,total_relevant,k),
                AP=ap_at_k(labels,total_relevant,k), F1=f1_at_k(labels,total_relevant,k),
                RR=rr_at_k(labels,k), NDCG=ndcg_at_k(labels,total_relevant,k))


def aggregate(per_query: list[dict]) -> dict[str,float]:
    return {'MAP': sum(r['AP'] for r in per_query)/len(per_query),
            'MRR': sum(r['RR'] for r in per_query)/len(per_query)} if per_query else {'MAP':0.0, 'MRR':0.0}


def evaluate_qrels(query: str, ranked: list[tuple[str,float]], qrels:dict[str,int], k=10):
    """Condensed ranking: unjudged products ignored, judged-but-unretrieved
    products count in recall denominators. Missing top-K slots counted as zero.
    This is an explicit evaluation choice due to partial judgments.
    """
    judged = [(pid,score) for pid,score in ranked if pid in qrels]
    labels = [int(qrels[pid]) for pid,_ in judged]
    metrics = evaluate(labels, sum(qrels.values()), k)
    return metrics, judged
