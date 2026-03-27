"""
run_evaluation.py — Script đánh giá tổng hợp cả Module 1 và Module 2

Sử dụng:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --data-dir data/annotated --mode partial --cluster
    python scripts/run_evaluation.py --mode exact
    python scripts/run_evaluation.py --cluster
"""
import os
import sys
import argparse

# Đảm bảo project root trong sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.evaluation.evaluator import (
    evaluate_by_dataset,
    evaluate_clustering_pipeline,
)

DEFAULT_DATA_DIR    = os.path.join(PROJECT_ROOT, "dataset", "annotated")
DEFAULT_CLUSTER_PATH = None  # sẽ dùng ATTRIBUTE_CLUSTERS_PATH từ nlacp.paths
DEFAULT_POLICY_PATH  = None  # sẽ dùng POLICY_DATASET_PATH từ nlacp.paths


def main():
    parser = argparse.ArgumentParser(
        description="NLACP — Unified F1 Evaluation (Module 1 + Module 2)"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"Thư mục chứa file annotated JSON (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--mode", default="partial", choices=["exact", "partial", "overlap"],
        help="Mode so sánh cho Module 1 (default: partial)"
    )
    parser.add_argument(
        "--cluster", action="store_true",
        help="Bật đánh giá Module 2 (Attribute Clustering)"
    )
    parser.add_argument(
        "--cluster-path", default=DEFAULT_CLUSTER_PATH,
        help="Override đường dẫn file attribute_clusters.json"
    )
    parser.add_argument(
        "--policy-path", default=DEFAULT_POLICY_PATH,
        help="Override đường dẫn file policy_dataset.json"
    )
    parser.add_argument(
        "--no-module1", action="store_true",
        help="Bỏ qua Module 1 (chỉ chạy Module 2)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="In chi tiết các câu sai"
    )
    args = parser.parse_args()

    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + "  NLACP — F1 Evaluation Report".center(68) + "║")
    print("╚" + "═"*68 + "╝")

    # ── Module 1 ────────────────────────────────────────────────────
    if not args.no_module1:
        if not os.path.isdir(args.data_dir):
            print(f"\n[WARN] data-dir không tồn tại: {args.data_dir}")
            print("       Bỏ qua Module 1. Chạy với --no-module1 để tắt cảnh báo này.")
        else:
            evaluate_by_dataset(
                args.data_dir,
                mode=args.mode,
                verbose=args.verbose,
                leave_one_out=True,
            )

    # ── Module 2 ────────────────────────────────────────────────────
    if args.cluster:
        evaluate_clustering_pipeline(
            cluster_path=args.cluster_path,
            policy_path=args.policy_path,
        )
    elif not args.no_module1:
        print("\n[INFO] Để đánh giá Module 2 (Clustering), thêm flag: --cluster")
        print("       Ví dụ: python scripts/run_evaluation.py --cluster\n")


if __name__ == "__main__":
    main()
