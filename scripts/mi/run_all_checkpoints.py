"""
Run MI experiments across all available checkpoints.

Discovers TRM and Transformer checkpoints, runs selected experiments on
each, and produces per-checkpoint + global aggregated results.

Usage:
    # Run all experiments on all checkpoints
    python3 scripts/mi/run_all_checkpoints.py --ckpt-dir checkpoints/

    # Run only experiments 1 and 7
    python3 scripts/mi/run_all_checkpoints.py --ckpt-dir checkpoints/ --experiments 1 7

    # Limit to 2 TRM checkpoints (for testing)
    python3 scripts/mi/run_all_checkpoints.py --ckpt-dir checkpoints/ --max-trm-ckpts 2
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_experiment(
    exp_script: str,
    args: list[str],
    label: str,
) -> bool:
    """Run an experiment script as a subprocess.

    Returns True if successful, False otherwise.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "mi" / exp_script)] + args
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=7200,  # 2h timeout per experiment
        )
        if result.returncode != 0:
            logger.error("%s FAILED (exit %d):\n%s", label, result.returncode, result.stderr[-1000:])
            return False
        logger.info("%s completed successfully", label)
        return True
    except subprocess.TimeoutExpired:
        logger.error("%s TIMED OUT", label)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MI experiments on all checkpoints")
    parser.add_argument("--ckpt-dir", default="checkpoints/",
                       help="Root checkpoint directory")
    parser.add_argument("--experiments", nargs="+", type=int,
                       default=[1, 2, 3, 4, 5, 6, 7, 8],
                       help="Which experiments to run (default: all 1-8)")
    parser.add_argument("--output-base", default=None,
                       help="Base output dir (default: outputs/mi/{domain})")
    parser.add_argument("--max-trm-ckpts", type=int, default=None,
                       help="Max TRM checkpoints (for testing)")
    parser.add_argument("--max-trans-ckpts", type=int, default=None,
                       help="Max Transformer checkpoints (for testing)")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Num samples for experiments")
    parser.add_argument("--domain", default="sudoku", choices=["sudoku", "arc"],
                       help="Domain: sudoku (MLP-T) or arc (attention TRM)")
    parser.add_argument("--arc-dataset-dir", default=None,
                       help="ARC dataset dir required when --domain arc")
    args = parser.parse_args()

    model_type   = "arc_trm" if args.domain == "arc" else "original_trm"
    output_base  = args.output_base or f"outputs/mi/{args.domain}"
    ckpt_dir = str(Path(args.ckpt_dir).resolve())
    results = {}

    for exp_num in args.experiments:
        logger.info("=" * 70)
        logger.info("EXPERIMENT %d", exp_num)
        logger.info("=" * 70)

        exp_args: list[str] = []

        if exp_num == 1:
            exp_args = [
                "--trm-ckpt-dir", ckpt_dir,
                "--num-samples", str(args.num_samples),
                "--output-dir", f"{output_base}/exp1",
            ]
            if args.domain:
                exp_args += ["--domain", args.domain]
            results[exp_num] = run_experiment(
                "causal_interventions.py", exp_args, "Exp 1"
            )

        elif exp_num == 2:
            exp_args = [
                "--trm-ckpt-dir", ckpt_dir,
                "--model-type", model_type,
                "--num-samples", str(args.num_samples),
                "--output-dir", f"{output_base}/exp2",
            ]
            if args.domain:
                exp_args += ["--domain", args.domain]
            if args.domain == "arc" and args.arc_dataset_dir:
                exp_args += ["--arc-dataset-dir", args.arc_dataset_dir]
            results[exp_num] = run_experiment(
                "representation_similarity.py", exp_args, "Exp 2"
            )

        elif exp_num == 3:
            exp_args = [
                "--trm-ckpt-dir", ckpt_dir,
                "--trans-ckpt-dir", ckpt_dir,
                "--num-samples", str(args.num_samples),
                "--output-dir", f"{output_base}/exp3",
            ]
            if args.domain:
                exp_args += ["--domain", args.domain]
            results[exp_num] = run_experiment(
                "information_bottleneck.py", exp_args, "Exp 3"
            )

        elif exp_num == 4:
            exp_args = [
                "--trm-ckpt-dir", ckpt_dir,
                "--trans-ckpt-dir", ckpt_dir,
                "--num-samples", str(args.num_samples),
                "--output-dir", f"{output_base}/exp4",
            ]
            if args.domain:
                exp_args += ["--domain", args.domain]
            results[exp_num] = run_experiment(
                "intrinsic_dimensionality.py", exp_args, "Exp 4"
            )

        elif exp_num == 5:
            exp_args = [
                "--trm-ckpt-dir", ckpt_dir,
                "--trans-ckpt-dir", ckpt_dir,
                "--num-samples", str(args.num_samples),
                "--output-dir", f"{output_base}/exp5",
            ]
            results[exp_num] = run_experiment(
                "ood_blanks_sweep.py", exp_args, "Exp 5"
            )

        elif exp_num == 6:
            exp_args = [
                "--trm-ckpt-dir", ckpt_dir,
                "--model-type", model_type,
                "--num-samples", str(args.num_samples),
                "--output-dir", f"{output_base}/exp6",
            ]
            if args.domain:
                exp_args += ["--domain", args.domain]
            if args.domain == "arc" and args.arc_dataset_dir:
                exp_args += ["--arc-dataset-dir", args.arc_dataset_dir]
            results[exp_num] = run_experiment(
                "superposition_analysis.py", exp_args, "Exp 6"
            )

        elif exp_num == 7:
            exp_args = [
                "--trm-ckpt-dir", ckpt_dir,
                "--model-type", model_type,
                "--num-samples", str(args.num_samples),
                "--output-dir", f"{output_base}/exp7",
            ]
            if args.domain == "arc" and args.arc_dataset_dir:
                exp_args += ["--arc-dataset-dir", args.arc_dataset_dir]
            if args.domain:
                exp_args += ["--domain", args.domain]
            results[exp_num] = run_experiment(
                "token_mixer_dissection.py", exp_args, "Exp 7"
            )

        elif exp_num == 8:
            exp_args = [
                "--trm-ckpt-dir", ckpt_dir,
                "--model-type", model_type,
                "--num-samples", str(args.num_samples),
                "--output-dir", f"{output_base}/exp8",
            ]
            if args.domain == "arc" and args.arc_dataset_dir:
                exp_args += ["--arc-dataset-dir", args.arc_dataset_dir]
            if args.domain:
                exp_args += ["--domain", args.domain]
            results[exp_num] = run_experiment(
                "circuit_discovery.py", exp_args, "Exp 8"
            )

        else:
            logger.warning("Unknown experiment number: %d", exp_num)
            continue

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for exp_num, success in results.items():
        status = "OK" if success else "FAILED"
        logger.info("  Exp %d: %s", exp_num, status)


if __name__ == "__main__":
    main()
