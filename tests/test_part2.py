import unittest
from collections import Counter
import pandas as pd
from project_progress.part_2.search_engine import build_inverted_index, calculate_idf, normalize_tf, conjunctive_search, rank_documents
from project_progress.part_2.evaluation import precision_at_k, recall_at_k, ap_at_k, f1_at_k, rr_at_k, ndcg_at_k, aggregate

class Part2Tests(unittest.TestCase):
    def test_and_across_fields(self):
        small=pd.DataFrame([
            {'pid':'A','title':['cotton'],'description':['sweatshirt']},
            {'pid':'B','title':['cotton'],'description':['jeans']},
            {'pid':'C','title':['wool'],'description':['sweatshirt']},
        ])
        for f in ('brand','category','sub_category','seller','product_details'):small[f]=[[]]*3
        idx,tf,df=build_inverted_index(small)
        from project_progress.part_2.search_engine import stopword_set
        self.assertEqual(conjunctive_search('cotton sweatshirt',idx,stopword_set()),['A'])
        self.assertEqual(conjunctive_search('cotton banana',idx,stopword_set()),[])
        idf=calculate_idf(df,3)
        scored=rank_documents('cotton sweatshirt',idx,normalize_tf(tf),idf,stopword_set())
        self.assertEqual([x[0] for x in scored],['A'])

    def test_metrics_hand_checked(self):
        labels=[1,0,1,0,1]
        self.assertAlmostEqual(precision_at_k(labels,5),0.6)
        self.assertAlmostEqual(recall_at_k(labels,4,5),0.75)
        self.assertAlmostEqual(ap_at_k(labels,4,5),(1+2/3+3/5)/4)
        self.assertAlmostEqual(f1_at_k(labels,4,5),2*.6*.75/(.6+.75))
        self.assertAlmostEqual(rr_at_k(labels,5),1.0)
        self.assertGreaterEqual(ndcg_at_k(labels,4,5),0)
        self.assertLessEqual(ndcg_at_k(labels,4,5),1)
        self.assertEqual(aggregate([{'AP':.5,'RR':1},{'AP':.75,'RR':.5}]),{'MAP':.625,'MRR':.75})

if __name__=='__main__':unittest.main()
