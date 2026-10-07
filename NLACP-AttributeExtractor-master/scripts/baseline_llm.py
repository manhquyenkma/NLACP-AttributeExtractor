#!/usr/bin/env python
"""
scripts/baseline_llm.py — LLM Baseline cho ABAC Attribute Extraction

So sánh hệ thống rule-based với LLM zero-shot extraction.
Hỗ trợ: Google Gemini (miễn phí), OpenAI GPT-4.

Usage:
    # Gemini (miễn phí - khuyên dùng):
    set GEMINI_API_KEY=your_key_here
    python scripts/baseline_llm.py --provider gemini

    # OpenAI GPT-4:
    set OPENAI_API_KEY=your_key_here
    python scripts/baseline_llm.py --provider openai

    # Chỉ chạy trên VACT:
    python scripts/baseline_llm.py --provider gemini --dataset vact

Output: outputs/policies/{dataset}_llm.json → dùng unified_eval.py để so sánh.
"""

import json
import os
import sys
import time
import argparse
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Prompt template ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in Attribute-Based Access Control (ABAC) policy analysis.
Given a natural language access control policy sentence, extract the following components:

1. **subjects**: List of subject entities (who is performing the action). Each subject has:
   - "role": the role/entity name (e.g., "doctor", "student", "administrator")
   - "entity_type": one of "user", "role", "group", "system"

2. **actions**: List of actions being performed. Each action has:
   - "verb": the verb lemma (e.g., "view", "create", "delete")
   - "operation": CRUD mapping: "read", "create", "update", "delete", or "execute"

3. **resources**: List of objects/resources being acted upon. Each resource has:
   - "label": the full resource name (e.g., "medical records", "patient data")

4. **environments**: List of environmental conditions. Each environment has:
   - "trigger_phrase": the exact text span from the sentence (e.g., "during business hours", "within the hospital")
   - "env_type": one of "temporal", "spatial_physical", "spatial_network", "spatial_device", "conditional"

Return ONLY valid JSON with these exact field names. If a component is not present, use an empty list [].
Do NOT include any explanation or markdown formatting — just the raw JSON object."""

USER_PROMPT_TEMPLATE = """Extract ABAC components from this policy sentence:

"{sentence}"

Return JSON with fields: subjects, actions, resources, environments."""


# ── API Callers ──────────────────────────────────────────────────────

def call_gemini(sentence: str, api_key: str, model: str = "gemini-2.0-flash") -> dict:
    """Call Google Gemini API."""
    import urllib.request
    import urllib.error

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(sentence=sentence)}
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_json_response(text)
    except Exception as e:
        print(f"  [ERROR] Gemini API: {e}")
        return {}


def call_openai(sentence: str, api_key: str, model: str = "gpt-4o-mini") -> dict:
    """Call OpenAI API."""
    import urllib.request
    import urllib.error

    url = "https://api.openai.com/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(sentence=sentence)}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    })
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"]
        return _parse_json_response(text)
    except Exception as e:
        print(f"  [ERROR] OpenAI API: {e}")
        return {}


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"  [WARN] Could not parse JSON: {text[:200]}...")
        return {}


# ── Dataset loading ──────────────────────────────────────────────────

DATASETS = {
    "vact": {
        "input": os.path.join(PROJECT_ROOT, "vact_acp.txt"),
        "gold": os.path.join(PROJECT_ROOT, "outputs", "policies", "vact_gold.json"),
        "output": os.path.join(PROJECT_ROOT, "outputs", "policies", "vact_llm.json"),
    },
    "itrust": {
        "input": os.path.join(PROJECT_ROOT, "itrust_acp.txt"),
        "gold": os.path.join(PROJECT_ROOT, "outputs", "policies", "itrust_gold.json"),
        "output": os.path.join(PROJECT_ROOT, "outputs", "policies", "itrust_llm.json"),
    },
    "edocument": {
        "input": os.path.join(PROJECT_ROOT, "edocument_acp.txt"),
        "gold": os.path.join(PROJECT_ROOT, "outputs", "policies", "edocument_gold.json"),
        "output": os.path.join(PROJECT_ROOT, "outputs", "policies", "edocument_llm.json"),
    },
}


def load_sentences(path: str) -> list:
    """Load sentences from text file (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return lines


def format_llm_output(policy_id: int, sentence: str, extracted: dict) -> dict:
    """Convert LLM output to unified evaluation format."""
    policy = {
        "id": policy_id,
        "sentence": sentence,
        "subjects": extracted.get("subjects", []),
        "actions": extracted.get("actions", []),
        "resources": extracted.get("resources", []),
        "environments": extracted.get("environments", []),
    }
    return policy


# ── Main ─────────────────────────────────────────────────────────────

def run_extraction(dataset_name: str, provider: str, api_key: str):
    """Run LLM extraction on a dataset."""
    ds = DATASETS[dataset_name]
    sentences = load_sentences(ds["input"])
    
    caller = call_gemini if provider == "gemini" else call_openai
    model_name = "gemini-2.0-flash" if provider == "gemini" else "gpt-4o-mini"
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()} ({len(sentences)} sentences)")
    print(f"Provider: {provider} ({model_name})")
    print(f"{'='*60}")
    
    policies = []
    errors = 0
    
    for i, sentence in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] {sentence[:80]}...")
        
        extracted = caller(sentence, api_key, model_name)
        
        if not extracted:
            errors += 1
            extracted = {"subjects": [], "actions": [], "resources": [], "environments": []}
        
        policy = format_llm_output(i + 1, sentence, extracted)
        policies.append(policy)
        
        # Rate limiting
        time.sleep(0.5 if provider == "gemini" else 0.2)
    
    # Save output
    output = {"policies": policies}
    os.makedirs(os.path.dirname(ds["output"]), exist_ok=True)
    with open(ds["output"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(policies)} policies to {ds['output']}")
    print(f"  Errors: {errors}/{len(sentences)}")
    
    return ds["output"]


def run_eval(dataset_name: str):
    """Run unified_eval.py comparison for a dataset."""
    ds = DATASETS[dataset_name]
    llm_path = ds["output"]
    gold_path = ds["gold"]
    
    if not os.path.exists(llm_path):
        print(f"  [SKIP] LLM output not found: {llm_path}")
        return
    
    # Import eval functions
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    from unified_eval import run as eval_run
    
    fmt = {"vact": "vact", "itrust": "itrust", "edocument": "edoc"}[dataset_name]
    
    res, errs = eval_run(dataset_name.upper() + " (LLM)", fmt, gold_path, llm_path)
    
    print(f"\n{'─'*60}")
    print(f"LLM Baseline Results — {dataset_name.upper()}")
    print(f"{'─'*60}")
    print(f"  Subject:  {res['Subject']:.4f}")
    print(f"  Action:   {res['Action']:.4f}")
    print(f"  Object:   {res['Object']:.4f}")
    print(f"  Env:      {res['Env']:.4f}")
    print(f"  Overall:  {res['Overall']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Baseline for ABAC extraction")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini",
                        help="LLM provider (default: gemini)")
    parser.add_argument("--dataset", choices=["vact", "itrust", "edocument", "all"], default="all",
                        help="Dataset to evaluate (default: all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation (skip extraction)")
    args = parser.parse_args()
    
    # Get API key
    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key and not args.eval_only:
        env_var = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"
        print(f"[ERROR] API key not found. Set environment variable: {env_var}")
        print(f"  Example: set {env_var}=your_key_here")
        sys.exit(1)
    
    datasets = ["vact", "itrust", "edocument"] if args.dataset == "all" else [args.dataset]
    
    for ds_name in datasets:
        if not args.eval_only:
            run_extraction(ds_name, args.provider, api_key)
        run_eval(ds_name)
    
    print(f"\n{'='*60}")
    print("Done! Compare these results with the rule-based pipeline:")
    print("  python scripts/unified_eval.py")
    print(f"{'='*60}")
