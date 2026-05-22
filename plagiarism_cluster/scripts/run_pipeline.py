#!/usr/bin/env python3
"""
scripts/run_pipeline.py
=======================
CLI entry-point: run the full pipeline on a data directory.

Usage:
    python scripts/run_pipeline.py --data-dir ./data --output-dir ./pan_output
    python scripts/run_pipeline.py --data-dir ./data --evaluate pan2011
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="PlagiarismShield CLI")
    parser.add_argument("--data-dir",   required=True,  help="Directory containing dataset zips")
    parser.add_argument("--output-dir", default=None,   help="Directory for PAN XML output files")
    parser.add_argument("--evaluate",   default=None,   help="Evaluate against: pan2011 | pan2015")
    parser.add_argument("--eps",        type=float, default=0.2,  help="DBSCAN eps (default 0.2)")
    parser.add_argument("--threshold",  type=float, default=0.80, help="Span similarity threshold")
    parser.add_argument("--load-train", action="store_true", help="Include PAN25 train split (23 GB)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data-dir not found: {args.data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  PlagiarismShield — Local Pipeline")
    print("=" * 60)

    from pipeline_core.pipeline import PlagiarismPipeline
    from ingestion.data_loader  import DataLoader

    loader  = DataLoader(args.data_dir, load_train=args.load_train)
    corpus  = loader.load_all()

    if not corpus:
        print("No documents found. Check --data-dir and dataset zips.")
        sys.exit(1)

    pp = PlagiarismPipeline(eps=args.eps, span_threshold=args.threshold)

    if args.evaluate:
        print(f"\nEvaluating against {args.evaluate} ground truth...")
        metrics = pp.evaluate_against_gt(args.data_dir, dataset=args.evaluate)
        print("\n── Evaluation Results ──────────────────────")
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")
    else:
        print(f"\nRunning pipeline on {len(corpus)} documents...")
        summary = pp.run_on_dataset(args.data_dir, output_dir=args.output_dir)
        print("\n── Summary ─────────────────────────────────")
        for k, v in summary.items():
            if k != "cluster_sizes":
                print(f"  {k:15s}: {v}")
        if args.output_dir:
            print(f"\n  PAN XML files written to: {args.output_dir}")

    print("\nDone.")

if __name__ == "__main__":
    main()
