import argparse
import json
import subprocess
from collections import OrderedDict
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from .fasta import fasta_iter
from .generate_coverage import calculate_coverage
from .generate_kmer import generate_feature_mapping, generate_kmer_features_from_fasta

DEFAULT_EDGE = 75
DEFAULT_CONTIG_THRESHOLD = 1000
DEFAULT_LENGTH_THRESHOLD = 1500
DEFAULT_KMER_LEN = 4


def old_calculate_coverage(depth_stream, bam_file, edge=DEFAULT_EDGE, contig_threshold=DEFAULT_CONTIG_THRESHOLD):
    contigs = []
    mean_coverage = []

    for contig_name, lines in groupby(depth_stream, lambda ell: ell.split("\t", 1)[0]):
        lengths = []
        values = []
        for line in lines:
            line_split = line.strip().split("\t")
            length = int(line_split[2]) - int(line_split[1])
            value = int(float(line_split[3]))
            lengths.append(length)
            values.append(value)

        depth_value = np.zeros(sum(lengths), dtype=int)
        s = 0
        for ell, v in zip(lengths, values):
            depth_value[s : s + ell] = v
            s += ell

        if len(depth_value) < contig_threshold:
            continue

        depth_value_ = depth_value[edge:-edge]
        mean_coverage.append(depth_value_.mean())
        contigs.append(contig_name)

    return pd.DataFrame({f"{bam_file}_cov": mean_coverage}, index=contigs)


def old_generate_kmer_features_from_fasta(
    fasta_file,
    length_threshold=DEFAULT_LENGTH_THRESHOLD,
    kmer_len=DEFAULT_KMER_LEN,
):
    kmer_dict, nr_features = generate_feature_mapping(kmer_len)
    composition = OrderedDict()
    seq_len = {}

    for h, seq in fasta_iter(fasta_file):
        if len(seq) < length_threshold:
            continue
        seq_len[h] = len(seq)
        norm_seq = str(seq).upper()
        kmers = [
            kmer_dict[norm_seq[i : i + kmer_len]]
            for i in range(len(norm_seq) - kmer_len + 1)
            if norm_seq[i : i + kmer_len] in kmer_dict
        ]
        composition[h] = np.bincount(np.array(kmers, dtype=np.int64), minlength=nr_features)

    df = pd.DataFrame.from_dict(composition, orient="index", dtype=float)
    df = df.apply(lambda x: x + 1e-5)
    df = df.div(df.sum(axis=1), axis=0)
    df_len = pd.DataFrame.from_dict(seq_len, orient="index")
    return df, df_len


def compare_frames(old_df, new_df, label):
    old_sorted = old_df.sort_index().sort_index(axis=1)
    new_sorted = new_df.sort_index().sort_index(axis=1)

    only_old_rows = sorted(set(old_sorted.index) - set(new_sorted.index))
    only_new_rows = sorted(set(new_sorted.index) - set(old_sorted.index))
    only_old_cols = sorted(set(old_sorted.columns) - set(new_sorted.columns))
    only_new_cols = sorted(set(new_sorted.columns) - set(old_sorted.columns))

    common_rows = old_sorted.index.intersection(new_sorted.index)
    common_cols = old_sorted.columns.intersection(new_sorted.columns)

    if len(common_rows) and len(common_cols):
        old_common = old_sorted.loc[common_rows, common_cols]
        new_common = new_sorted.loc[common_rows, common_cols]
        delta = (new_common - old_common).abs()
        max_abs = float(delta.to_numpy().max())
        exact_equal = bool(np.array_equal(old_common.to_numpy(), new_common.to_numpy(), equal_nan=True))
    else:
        max_abs = float("nan")
        exact_equal = (
            len(only_old_rows) == 0
            and len(only_new_rows) == 0
            and len(only_old_cols) == 0
            and len(only_new_cols) == 0
        )

    return {
        "name": label,
        "exact_equal": exact_equal,
        "max_abs_diff": max_abs,
        "rows_only_in_old": only_old_rows,
        "rows_only_in_new": only_new_rows,
        "cols_only_in_old": only_old_cols,
        "cols_only_in_new": only_new_cols,
    }


def load_depth_lines(bam=None, depth_bga=None):
    if depth_bga:
        return Path(depth_bga).read_text().splitlines(keepends=True)

    if not bam:
        raise ValueError("Either --bam or --depth-bga must be provided for coverage comparison.")

    completed = subprocess.run(
        ["bedtools", "genomecov", "-bga", "-ibam", bam],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.splitlines(keepends=True)


def run_original_preprocessing(fasta, depth_lines, bam_label):
    kmer, length = old_generate_kmer_features_from_fasta(fasta)
    cov = old_calculate_coverage(iter(depth_lines), bam_label)
    return kmer, length, cov


def run_optimized_preprocessing(fasta, depth_lines, bam_label):
    kmer, length = generate_kmer_features_from_fasta(fasta)
    cov = calculate_coverage(iter(depth_lines), bam_label)
    return kmer, length, cov


def run_compare(fasta, output_dir, bam=None, depth_bga=None, method="both"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    depth_lines = load_depth_lines(bam=bam, depth_bga=depth_bga)
    bam_label = bam if bam else "depth_bga_input"
    summary = {
        "parameters": {
            "fasta": str(fasta),
            "bam": bam,
            "depth_bga": depth_bga,
            "edge": DEFAULT_EDGE,
            "contig_threshold": DEFAULT_CONTIG_THRESHOLD,
            "length_threshold": DEFAULT_LENGTH_THRESHOLD,
            "kmer_len": DEFAULT_KMER_LEN,
            "method": method,
        },
        "comparisons": [],
    }

    old_kmer = old_len = old_cov = None
    new_kmer = new_len = new_cov = None

    if method in {"original", "both"}:
        old_kmer, old_len, old_cov = run_original_preprocessing(fasta, depth_lines, bam_label)
        old_kmer.to_csv(output_path / "kmer_original.csv")
        old_len.to_csv(output_path / "length_original.csv")
        old_cov.to_csv(output_path / "coverage_original.csv")

    if method in {"optimized", "both"}:
        new_kmer, new_len, new_cov = run_optimized_preprocessing(fasta, depth_lines, bam_label)
        new_kmer.to_csv(output_path / "kmer_optimized.csv")
        new_len.to_csv(output_path / "length_optimized.csv")
        new_cov.to_csv(output_path / "coverage_optimized.csv")

    if method == "both":
        summary["comparisons"] = [
            compare_frames(old_kmer, new_kmer, "kmer"),
            compare_frames(old_len, new_len, "length"),
            compare_frames(old_cov, new_cov, "coverage"),
        ]

    (output_path / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run original and/or optimized LorBin preprocessing and compare outputs."
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output directory for comparison artifacts")
    parser.add_argument("--bam", default=None, help="Input BAM file; used to call bedtools genomecov -bga")
    parser.add_argument("--depth-bga", default=None, help="Optional precomputed bedtools genomecov -bga output")
    parser.add_argument(
        "--method",
        choices=["original", "optimized", "both"],
        default="both",
        help="Which preprocessing implementation(s) to run",
    )
    args = parser.parse_args()

    if not args.bam and not args.depth_bga:
        parser.error("Provide either --bam or --depth-bga for coverage comparison")

    return args


def main():
    args = parse_args()
    summary = run_compare(
        fasta=args.fasta,
        output_dir=args.output,
        bam=args.bam,
        depth_bga=args.depth_bga,
        method=args.method,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
