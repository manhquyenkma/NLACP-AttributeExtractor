ABAC Policy NLP Extraction
Based on: Alohaly et al. (2019). Automated extraction of attributes from NLACP. Cybersecurity 2:2

==================================================
  PROJECT STRUCTURE
==================================================

project/
 main.py                      # Full pipeline entry point (5 modules)
 test_pipeline.py             # Auto-test with 5 sample sentences
 annotate_helper.py           # Interactive annotation tool
 
 src/
   nlp_engine.py              # Module 1 entry + loop
   relation_candidate.py      # Module 1: NLP + Top-5 dependency patterns
   data_type_infer.py         # Module 5: NER-based data type inference
   dataset_builder.py         # Dataset read/write
   cnn_classifier.py          # FIX 5: CNN classifier (use after 200+ annotated)

 mining/
   attribute_cluster.py       # Module 2: DBSCAN + auto-tune eps (TF-IDF, FIX 2)
   attribute_cluster_glove.py # FIX 3: DBSCAN + GloVe (swap after en_core_web_md)
   namespace_hierarchy.py     # Module 3: ABAC hierarchical namespace

 dataset/
   policy_dataset.json        # Extracted policies (output of Module 1)
   attribute_clusters.json    # Clusters (output of Module 2)
   namespace_hierarchy.json   # ABAC hierarchy (output of Module 3)
   annotation_template.json   # Template to annotate 851 sentences
   annotated_corpus.json      # Annotated corpus (for CNN training)

==================================================
  HOW TO RUN
==================================================

[Full pipeline]
  python main.py

[Quick test with 5 sample sentences]
  python test_pipeline.py

[Annotate sentences manually]
  python annotate_helper.py

==================================================
  ROADMAP
==================================================

This week  (FIX 1+2 DONE):
  - Top-5 dependency patterns (amod/compound/acl/prep/nummod)
  - Auto-tune eps with k-distance graph, min_samples=2

Next week  (FIX 3):
  - python -m spacy download en_core_web_md
  - Copy mining/attribute_cluster_glove.py -> mining/attribute_cluster.py
  - GloVe 300d vectors instead of TF-IDF

Parallel (ongoing):
  - Collect 851 annotated sentences using annotate_helper.py
  - Sources: iTrust, IBM App, CyberChair, manual
  - Target: 20-30 sentences/person/week

After 200+ annotated sentences (FIX 5):
  - pip install torch
  - python src/cnn_classifier.py --train --data dataset/annotated_corpus.json --type subject
  - python src/cnn_classifier.py --train --data dataset/annotated_corpus.json --type object

==================================================
  ATTRIBUTE FORMAT (policy_dataset.json)
==================================================

Each attribute now has 4 dimensions (vs bài báo 5 dimensions):
  name     : modifier text (e.g. "senior")
  value    : element it modifies (e.g. "nurse")
  category : "subject" or "object"             [Module 4]
  dep      : dependency relation used           [Module 1]
  data_type: "string" | "integer" | "datetime" [Module 5]

Missing vs paper:
  short_name : suggested name (e.g. "rank") -> needs 851 corpus + CNN
  namespace  : hierarchical assignment      -> Module 3 in progress

==================================================
  REQUIREMENTS
==================================================

  pip install spacy scikit-learn nltk
  python -m spacy download en_core_web_sm

  (Next week):
  python -m spacy download en_core_web_md

  (After data collection):
  pip install torch

==================================================
  REFERENCES
==================================================

  Alohaly et al. (2019). Automated extraction of attributes from
  natural language ABAC policies. Cybersecurity 2:2.
  https://doi.org/10.1186/s42400-018-0019-2