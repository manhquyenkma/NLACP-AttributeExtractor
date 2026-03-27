"""
nlacp/paths.py — Centralized path definitions for the project.
All modules should import paths from here to avoid hardcoding.
"""
import os

# nlacp/ -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RELATION_CANDIDATE_PATH = os.path.join(DATASET_DIR, "relation_candidate.json")

# Outputs (Now merged into dataset/)
POLICY_DATASET_PATH     = os.path.join(DATASET_DIR, "policy_dataset.json")
ATTRIBUTE_CLUSTERS_PATH = os.path.join(DATASET_DIR, "attribute_clusters.json")
NAMESPACE_HIERARCHY_PATH = os.path.join(DATASET_DIR, "namespace_hierarchy.json")

# Namespace Constants
NS_ENV_TIME      = "env:time"
NS_ENV_LOC       = "env:location"
