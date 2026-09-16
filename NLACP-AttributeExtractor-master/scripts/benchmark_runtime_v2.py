import time
import json
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath('.'))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.utils.nlp_utils import get_spacy_model
from nlacp.extraction.env_extractor import extract_env_attributes
from nlacp.extraction.subject_extractor import extract_subjects_with_logical_op
from nlacp.extraction.action_extractor import extract_actions_with_logical_op
from nlacp.extraction.resource_extractor import extract_resources
from nlacp.pipeline.pipeline import process_sentence

def benchmark_pipeline(sentences: list, n_values: list = [10, 25, 53, 100, 153]):
    # 1. Warm-up spaCy
    nlp = get_spacy_model()
    
    # WARM UP EVERY STAGE PROPERLY (1 sentence)
    warm_up_s = sentences[0]
    _ = extract_env_attributes(warm_up_s)
    doc = nlp(warm_up_s)
    _ = extract_subjects_with_logical_op(warm_up_s, doc)
    _ = extract_actions_with_logical_op(warm_up_s, doc)
    _ = extract_resources(warm_up_s, doc)
    _ = process_sentence(warm_up_s)
    
    results = []
    
    for n in n_values:
        subset = sentences[:n]
        
        # Average over 3 runs for stability, especially for small N
        num_runs = 3 if n <= 25 else 1
        
        t1_total, t2_total, tf_total = 0, 0, 0
        
        for _run in range(num_runs):
            t0 = time.perf_counter()
            # Stage 1: env extraction riêng
            for s in subset:
                _ = extract_env_attributes(s)
            t1 = time.perf_counter()
        
            # Stage 2: subject+action+resource
            for s in subset:
                doc = nlp(s)
                _ = extract_subjects_with_logical_op(s, doc)
                _ = extract_actions_with_logical_op(s, doc)
                _ = extract_resources(s, doc)
            t2 = time.perf_counter()
        
            # Full pipeline
            t3 = time.perf_counter()
            for s in subset:
                _ = process_sentence(s)
            t4 = time.perf_counter()
            
            t1_total += (t1 - t0)
            t2_total += (t2 - t1)
            tf_total += (t4 - t3)
        
        results.append({
            'n': n,
            'stage1_env_sec': t1_total / num_runs,
            'stage2_sao_sec': t2_total / num_runs,
            'full_pipeline_sec': tf_total / num_runs,
            'ms_per_sentence': (tf_total / num_runs) * 1000 / n,
        })
    return results

if __name__ == '__main__':
    with open('outputs/policies/vact_gold.json', 'r', encoding='utf-8') as f:
        vact_gold = json.load(f)
    with open('outputs/policies/itrust_gold.json', 'r', encoding='utf-8') as f:
        itrust_gold = json.load(f)
        
    sentences = [p['sentence'] for p in vact_gold['policies']] + [p['sentence'] for p in itrust_gold['policies']]
    
    results = benchmark_pipeline(sentences)
    
    print('=== R2-3 BENCHMARK COMPLETE (WITH WARMUP) ===')
    print('{:<5} | {:<14} | {:<14} | {:<14} | {:<12}'.format('N', 'Stage1_Env(s)', 'Stage2_SAO(s)', 'Full_Pipe(s)', 'ms/sentence'))
    print('-' * 65)
    for r in results:
        print('{:>5} | {:>14.4f} | {:>14.4f} | {:>14.4f} | {:>12.2f}'.format(r['n'], r['stage1_env_sec'], r['stage2_sao_sec'], r['full_pipeline_sec'], r['ms_per_sentence']))
