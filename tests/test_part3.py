"""Smoke tests for Part 3 using a small synthetic product catalog."""
import unittest

import pandas as pd

from project_progress.part_2.search_engine import FIELDS
from project_progress.part_3.part3_ranking import (
    Part3Ranker, _discount_norm, _stock_norm, _rating_norm,
)


class Part3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        products = pd.DataFrame([
            {"pid": "A", "title": "Blue cotton shirt", "average_rating": 4.8,
             "discount": 0.35, "out_of_stock": False},
            {"pid": "B", "title": "Cotton fashion", "average_rating": 2.0,
             "discount": "50% off", "out_of_stock": True},
            {"pid": "C", "title": "Red jeans", "average_rating": None,
             "discount": None, "out_of_stock": False},
        ])
        tokens = pd.DataFrame([
            {"pid": "A", "title": ["blue", "cotton", "shirt"],
             "description": ["blue", "cotton"]},
            {"pid": "B", "title": ["cotton", "fashion"],
             "description": ["blue", "cloth"]},
            {"pid": "C", "title": ["red", "jean"],
             "description": ["denim"]},
        ])
        for field in FIELDS:
            if field not in tokens:
                tokens[field] = [[] for _ in range(len(tokens))]
        cls.engine = Part3Ranker(products, tokens, stops=set())

    def test_shared_boolean_and(self):
        for method in ("tfidf_cosine", "bm25", "custom"):
            ranked = getattr(self.engine, method)("blue cotton")
            self.assertEqual({pid for pid, _ in ranked}, {"A", "B"})
        self.assertEqual(self.engine.candidates("missingterm"), [])

    def test_tfidf_cosine(self):
        ranked = self.engine.tfidf_cosine("blue cotton")
        self.assertTrue(all(0 <= score <= 1.000001 for _, score in ranked))
        self.assertEqual(ranked[0][0], "A")

    def test_bm25(self):
        ranked = self.engine.bm25("blue cotton")
        self.assertEqual(ranked[0][0], "A")

    def test_custom_score_and_normalization(self):
        self.assertAlmostEqual(_discount_norm({"discount": 0.35}), 0.35)
        self.assertAlmostEqual(_discount_norm({"discount": "35% off"}), 0.35)
        self.assertEqual(_stock_norm({"out_of_stock": True}), 0.0)
        self.assertEqual(_stock_norm({"out_of_stock": False}), 1.0)
        self.assertEqual(_rating_norm({"average_rating": None}), 0.5)
        ranked = self.engine.custom("blue cotton")
        self.assertEqual({pid for pid, _ in ranked}, {"A", "B"})

    def test_word2vec(self):
        self.engine.train_word2vec(vector_size=12, epochs=3)
        ranked = self.engine.word2vec_cosine("blue cotton")
        self.assertEqual({pid for pid, _ in ranked}, {"A", "B"})
        self.assertEqual(len(self.engine.results_table("blue cotton", "Word2Vec cosine")), 2)
        self.assertEqual(self.engine.word2vec_cosine("missingterm"), [])


if __name__ == "__main__":
    unittest.main()
