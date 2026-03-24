import sys
import os
import unittest
import json

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from experiments.cnn.cnn_classifier import RelationDataset
    HAS_CNN = True
except (ImportError, ModuleNotFoundError):
    HAS_CNN = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.relation_candidate import extract_relations, parse_sentence, generate_candidates
from nlacp.extraction.env_extractor import extract_env_attributes
from nlacp.normalization.category_identifier import identify_categories
from nlacp.normalization.namespace_assigner import assign_namespaces
from nlacp.extraction.short_name_suggester import suggest_short_names
from nlacp.normalization.data_type_infer import infer_data_type
from nlacp.pipeline.pipeline import process_sentence

class TestArchitectureUpgrade(unittest.TestCase):

    def test_1_direct_object_wins_over_pobj(self):
        # TEST 1
        s = "Doctors may read and write medical records during business hours within the hospital."
        tokens = parse_sentence(s)
        rel_data = extract_relations(s, tokens)
        
        self.assertEqual(rel_data["object"], "records", "Dobj 'records' should win")
        self.assertNotIn("hospital", rel_data["object"])
        self.assertNotIn("hours", rel_data["object"])

    def test_2_env_phrases_included_in_raw_extraction(self):
        # TEST 2: Alohaly 2019 — extract_relations() trả về TẤT CẢ pairs
        # (bao gồm env). Env chỉ bị loại ở bước pipeline/ABAC_extraction.
        s = "Doctors may read and write medical records during business hours within the hospital."
        tokens = parse_sentence(s)
        rel = extract_relations(s, tokens)
        
        attr_values = [a["value"] for a in rel["attributes"]]
        # Env pairs PHẢI có mặt ở raw extraction
        self.assertIn("hours", attr_values, "hours should be in raw extraction pairs")
        
        # Nhưng sau khi qua pipeline, env phải bị tách riêng
        result = process_sentence(s)
        sa_values = [a.get("value") for a in result["attributes"]]
        self.assertNotIn("hours", sa_values, "hours must be filtered from SA/OA by pipeline")

    def test_3_category_normalization(self):
        # TEST 3
        attrs = [
            {"category": "temporal", "sub_category": ""},
            {"category": "spatial", "sub_category": "physical"}
        ]
        norm = identify_categories(attrs, "")
        
        self.assertEqual(norm[0]["category"], "environment")
        self.assertEqual(norm[0]["sub_category"], "temporal")
        self.assertEqual(norm[1]["category"], "environment")
        self.assertEqual(norm[1]["sub_category"], "physical")

    def test_4_namespace_assignment(self):
        # TEST 4
        attrs = [
            {"category": "environment", "sub_category": "temporal", "short_name": "business_hour"},
            {"category": "environment", "sub_category": "spatial", "short_name": "hospital"}
        ]
        named = assign_namespaces(attrs, "", "")
        
        self.assertEqual(named[0]["namespace"], "env:time:business_hour")
        self.assertEqual(named[1]["namespace"], "env:location:hospital")

    def test_5_short_name_generation_env_attrs(self):
        # TEST 5
        attrs = [
            {"category": "temporal", "value": "during business hours"},
            {"category": "spatial", "value": "within the hospital"}
        ]
        shorted = suggest_short_names(attrs)
        
        self.assertEqual(shorted[0]["short_name"], "during_business_hour")
        self.assertEqual(shorted[1]["short_name"], "within_hospital")

    def test_6_category_aware_dtype(self):
        # TEST 6
        self.assertEqual(infer_data_type("during business hours", "environment", "temporal"), "datetime")
        self.assertEqual(infer_data_type("true"), "boolean")
        self.assertEqual(infer_data_type("100"), "integer")
        self.assertEqual(infer_data_type("within hospital", "environment", "spatial"), "string")

    def test_7_candidate_generation_schema(self):
        # TEST 7
        s = "An on-call senior nurse may change approved lab procedures."
        data = generate_candidates(s)
        
        self.assertIn("candidates", data)
        cands = data["candidates"]
        
        # Look for valid positive
        positives = [c for c in cands if c["valid"]]
        self.assertTrue(len(positives) > 0)
        
        # Look for valid negative (random combinations that aren't deps)
        negatives = [c for c in cands if not c["valid"]]
        self.assertTrue(len(negatives) > 0)
        
        # Schema checks
        self.assertIn("element", cands[0])
        self.assertIn("modifier", cands[0])
        self.assertIn("category", cands[0])
        self.assertIn("valid", cands[0])

    @unittest.skipIf(not HAS_TORCH or not HAS_CNN, "PyTorch or CNN module not available")
    def test_8_cnn_dataset_loader(self):
        # TEST 8
        dummy_corpus = {
            "sentences": [
                {
                    "subject": "nurse",
                    "object": "procedures",
                    "candidates": [
                        {"element": "nurse", "modifier": "senior", "valid": True, "category": "subject"},
                        {"element": "nurse", "modifier": "lab", "valid": False, "category": "subject"}
                    ]
                }
            ]
        }
        
        os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "policies"), exist_ok=True)
        dummy_path = os.path.join(PROJECT_ROOT, "outputs", "policies", "_dummy_test_corpus.json")
        with open(dummy_path, "w") as f:
            json.dump(dummy_corpus, f)
            
        dataset = RelationDataset(dummy_path, relation_type="subject")
        
        # 2 samples in the subject category
        self.assertEqual(len(dataset), 2)
        words1, label1 = dataset.samples[0]
        words2, label2 = dataset.samples[1]
        
        # First is valid
        self.assertEqual(label1, 1)
        # Second is invalid
        self.assertEqual(label2, 0)
        
        # Test __getitem__
        tensor_x, tensor_y = dataset[0]
        self.assertEqual(tensor_y.item(), 1.0)
        
        os.remove(dummy_path)

    def test_9_integration_full_process(self):
        # TEST 9
        s = "Doctors may approve medical records during business hours."
        res = process_sentence(s)
        
        # check basic structures
        self.assertEqual(res["subject"], "Doctors")
        self.assertIn("Update", res["actions"])
        self.assertEqual(res["object"], "records")
        
        # Environment should be separated
        self.assertTrue(len(res["environment"]) > 0)
        env0 = res["environment"][0]
        self.assertEqual(env0["type"], "temporal")
        self.assertIn("business", env0.get("full_value", ""))
        self.assertEqual(env0["data_type"], "time")
        self.assertTrue(env0["namespace"].startswith("env:time:"))
        
        # SA/OA should not contain "hours"
        sa_values = [a.get("value") for a in res["attributes"]]
        self.assertNotIn("hours", sa_values)
        self.assertNotIn("business", sa_values)


if __name__ == '__main__':
    unittest.main()
