#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_annotate_policy.py
======================
Dùng Claude API để sinh gold annotation (temporal_final / spatial_final)
cho toàn bộ 380 câu trong policy_dataset.json.

Output:
  - annotation_llm_gold.csv   → dùng với eval_policy_f1.py --csv
  - annotation_llm_gold.json  → backup full

Chạy:
  pip install anthropic
  python scripts/llm_annotate_policy.py
  python scripts/llm_annotate_policy.py --resume   # tiếp tục nếu bị ngắt giữa chừng
  python scripts/llm_annotate_policy.py --dry-run  # chạy 5 câu để test

Sau khi xong:
  python scripts/eval_policy_f1.py --csv --csv-path data/annotation_llm_gold.csv
"""

import json
import csv
import os
import sys
import time
import argparse
import re
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("[ERROR] Chưa cài anthropic. Chạy: pip install anthropic")
    sys.exit(1)

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
if (SCRIPT_DIR / "outputs").exists():
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = SCRIPT_DIR.parent
POLICY_PATH  = PROJECT_ROOT / "outputs" / "policies" / "policy_dataset.json"
OUT_DIR      = PROJECT_ROOT / "data"
OUT_CSV      = OUT_DIR / "annotation_llm_gold.csv"
OUT_JSON     = OUT_DIR / "annotation_llm_gold.json"
CHECKPOINT   = OUT_DIR / ".llm_annotate_checkpoint.json"

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert NLP annotator for Access Control Policy (ABAC) sentences.
Your task: extract TEMPORAL and SPATIAL environment conditions from a policy sentence.

DEFINITIONS:
- temporal: time-based constraints (business hours, weekdays, after 5pm, during shift, etc.)
- spatial: location-based constraints — either physical (hospital, lab, HQ), 
           network (VPN, intranet, company network), or device (secure terminal, workstation)

OUTPUT FORMAT (strict JSON, no explanation):
{
  "temporal": "<extracted phrase or empty string>",
  "spatial": "<extracted phrase or empty string>"
}

RULES:
1. Extract the MINIMAL phrase that conveys the condition (e.g. "during business hours", "within the hospital")
2. If multiple temporal/spatial values exist, join with "; " (e.g. "on weekdays; after login")
3. If no temporal constraint: return ""
4. If no spatial constraint: return ""
5. Do NOT include subject/action/object in the output
6. Prepositional phrase is part of the value (e.g. "within the hospital", not just "hospital")
7. Return ONLY valid JSON — no markdown, no explanation, no backticks"""

def make_user_prompt(sentence: str) -> str:
    return f'Sentence: "{sentence}"'

# ─── Claude API call ──────────────────────────────────────────────────────────

def call_claude(client: anthropic.Anthropic, sentence: str, retries: int = 3) -> dict:
    """Call Claude API and return {temporal, spatial}. Retry on failure."""
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": make_user_prompt(sentence)}]
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

            result = json.loads(raw)
            return {
                "temporal": str(result.get("temporal", "") or "").strip(),
                "spatial":  str(result.get("spatial",  "") or "").strip(),
            }
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error on attempt {attempt+1}: {e} | raw={raw[:80]}")
            if attempt < retries - 1:
                time.sleep(1)
        except anthropic.RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"  [WARN] Rate limit hit, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [WARN] API error on attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    # Fallback — empty annotation rather than crashing
    return {"temporal": "", "spatial": ""}

# ─── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}

def save_checkpoint(done: dict):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)

# ─── Main ─────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "ID", "Source", "Sentence",
    "temporal_gold", "spatial_gold",
    "temporal_OK", "spatial_OK",
    "temporal_final", "spatial_final",
    "note", "annotator", "status"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",   action="store_true", help="Tiếp tục từ checkpoint")
    parser.add_argument("--dry-run",  action="store_true", help="Chỉ chạy 5 câu để test")
    parser.add_argument("--delay",    type=float, default=0.3, help="Delay giữa các API call (giây)")
    parser.add_argument("--policy",   type=str, default=str(POLICY_PATH), help="Path tới policy_dataset.json")
    args = parser.parse_args()

    # Load dataset
    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"[ERROR] Không tìm thấy: {policy_path}")
        print("  Thử chạy từ project root hoặc truyền --policy <path>")
        sys.exit(1)

    with open(policy_path, encoding="utf-8") as f:
        data = json.load(f)
    policies = data["policies"]

    if args.dry_run:
        policies = policies[:5]
        print(f"[DRY RUN] Chỉ xử lý {len(policies)} câu đầu.")

    # Load checkpoint (nếu --resume)
    done = load_checkpoint() if args.resume else {}
    if done:
        print(f"[RESUME] Đã có {len(done)} câu trong checkpoint, bỏ qua...")

    # Init Claude client
    client = anthropic.Anthropic()  # dùng ANTHROPIC_API_KEY từ env

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total   = len(policies)

    print(f"\n{'='*60}")
    print(f"  LLM Auto-Annotate  |  {total} câu  |  model: claude-3-5-sonnet-20241022")
    print(f"{'='*60}")

    for i, policy in enumerate(policies, 1):
        pid      = str(policy["id"])
        sentence = policy["sentence"]

        # Skip nếu đã có trong checkpoint
        if pid in done:
            results.append(done[pid])
            continue

        print(f"  [{i:>3}/{total}] id={pid:>3} | {sentence[:60]}...")

        ann = call_claude(client, sentence)

        # --- Quick human-readable preview ---
        t_label = ann["temporal"] or "—"
        s_label = ann["spatial"]  or "—"
        print(f"           temporal={t_label!r:<30}  spatial={s_label!r}")

        row = {
            "ID":             f"P{int(pid):04d}",
            "Source":         "policy_dataset",
            "Sentence":       sentence,
            "temporal_gold":  ann["temporal"],
            "spatial_gold":   ann["spatial"],
            "temporal_OK":    "LLM",
            "spatial_OK":     "LLM",
            "temporal_final": ann["temporal"],   # ← gold cho eval_policy_f1.py
            "spatial_final":  ann["spatial"],    # ← gold cho eval_policy_f1.py
            "note":           "",
            "annotator":      "claude-3-5-sonnet-20241022",
            "status":         "llm_annotated"
        }

        results.append(row)
        done[pid] = row

        # Checkpoint mỗi 20 câu
        if i % 20 == 0:
            save_checkpoint(done)
            print(f"  [CHECKPOINT] Đã lưu {i}/{total} câu.")

        time.sleep(args.delay)

    # ─── Save CSV ────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # ─── Save JSON ───────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ─── Clean checkpoint ────────────────────────────────────────────────────
    if CHECKPOINT.exists() and not args.dry_run:
        CHECKPOINT.unlink()

    # ─── Stats ───────────────────────────────────────────────────────────────
    n_temporal = sum(1 for r in results if r["temporal_final"])
    n_spatial  = sum(1 for r in results if r["spatial_final"])
    n_both     = sum(1 for r in results if r["temporal_final"] and r["spatial_final"])
    n_empty    = sum(1 for r in results if not r["temporal_final"] and not r["spatial_final"])

    print(f"\n{'='*60}")
    print(f"  XONG!  {len(results)} câu đã annotate.")
    print(f"  Có temporal:   {n_temporal}")
    print(f"  Có spatial:    {n_spatial}")
    print(f"  Có cả hai:     {n_both}")
    print(f"  Không có env:  {n_empty}")
    print(f"\n  CSV  → {OUT_CSV}")
    print(f"  JSON → {OUT_JSON}")
    print(f"\n  Bước tiếp theo:")
    print(f"  1. Mở annotation_llm_gold.csv, review nhanh ~30-50 dòng")
    print(f"  2. Sửa cột temporal_final / spatial_final nếu LLM sai")
    print(f"  3. Chạy eval:")
    print(f"     python scripts/eval_policy_f1.py --csv --csv-path data/annotation_llm_gold.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
