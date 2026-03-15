import math
from collections import OrderedDict
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

from lorbin.generate_coverage import calculate_coverage
from lorbin.generate_kmer import (
    generate_feature_mapping,
    generate_kmer_features_from_fasta,
)


def old_calculate_coverage(depth_stream, bam_file, edge=75, contig_threshold=1000):
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

        cov_threshold = contig_threshold
        if len(depth_value) < cov_threshold:
            continue

        depth_value_ = depth_value[edge:-edge]
        mean_coverage.append(depth_value_.mean())
        contigs.append(contig_name)

    return pd.DataFrame({f"{bam_file}_cov": mean_coverage}, index=contigs)


def old_generate_kmer_features_from_fasta(fasta_file, length_threshold=1500, kmer_len=4):
    from lorbin.fasta import fasta_iter

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


@pytest.mark.parametrize(
    "depth_lines,edge,contig_threshold",
    [
        (
            [
                "c1\t0\t100\t2\n",
                "c1\t100\t200\t4\n",
                "c2\t0\t50\t1\n",
                "c2\t50\t120\t3\n",
            ],
            10,
            100,
        ),
        (
            [
                "tiny\t0\t120\t1\n",  # triggers empty trimmed region (120 - 2*75)
            ],
            75,
            100,
        ),
        (
            [
                "n1\t0\t500\t0\n",
                "n1\t500\t1500\t5\n",
                "n1\t1500\t2000\t2\n",
            ],
            25,
            1000,
        ),
    ],
)
def test_calculate_coverage_matches_pre_optimization(depth_lines, edge, contig_threshold):
    expected = old_calculate_coverage(iter(depth_lines), "sample.bam", edge=edge, contig_threshold=contig_threshold)
    actual = calculate_coverage(iter(depth_lines), "sample.bam", edge=edge, contig_threshold=contig_threshold)

    assert list(actual.index) == list(expected.index)
    assert list(actual.columns) == list(expected.columns)
    for a, b in zip(actual.iloc[:, 0].to_list(), expected.iloc[:, 0].to_list()):
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            assert math.isnan(a) and math.isnan(b)
        else:
            assert a == pytest.approx(b, rel=0, abs=0)


@pytest.mark.parametrize(
    "records,length_threshold,kmer_len",
    [
        ([('c1', 'ATGCGATGCGAT'), ('c2', 'TTTTAAAACCCC')], 1, 4),
        ([('short', 'ATGC'), ('long', 'ATGNNNNNCGATATAT')], 5, 3),
        ([('mixed', 'acgtACGTNNNNacgt')], 1, 4),
    ],
)
def test_generate_kmer_matches_pre_optimization(records, length_threshold, kmer_len):
    with NamedTemporaryFile('w', suffix='.fa', delete=False) as handle:
        for name, seq in records:
            handle.write(f'>{name}\n{seq}\n')
        fasta_path = Path(handle.name)

    try:
        expected_df, expected_len = old_generate_kmer_features_from_fasta(
            str(fasta_path),
            length_threshold=length_threshold,
            kmer_len=kmer_len,
        )
        actual_df, actual_len = generate_kmer_features_from_fasta(
            str(fasta_path),
            length_threshold=length_threshold,
            kmer_len=kmer_len,
        )

        pd.testing.assert_frame_equal(actual_df, expected_df, check_exact=False, rtol=0, atol=0)
        pd.testing.assert_frame_equal(actual_len, expected_len, check_exact=True)
    finally:
        fasta_path.unlink(missing_ok=True)
