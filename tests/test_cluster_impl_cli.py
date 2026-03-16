import sys

from lorbin.lorbin import parser_args


def test_bin_cluster_impl_original(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "LorBin",
            "bin",
            "-o",
            "out",
            "-fa",
            "input.fa",
            "-b",
            "input.bam",
            "--cluster_impl",
            "original",
        ],
    )
    args = parser_args()
    assert args.cluster_impl == "original"


def test_bin_cluster_impl_optimized_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["LorBin", "bin", "-o", "out", "-fa", "input.fa", "-b", "input.bam"],
    )
    args = parser_args()
    assert args.cluster_impl == "optimized"


def test_bin_recluster_impl_optimized(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "LorBin",
            "bin",
            "-o",
            "out",
            "-fa",
            "input.fa",
            "-b",
            "input.bam",
            "--recluster_impl",
            "optimized",
        ],
    )
    args = parser_args()
    assert args.recluster_impl == "optimized"


def test_bin_recluster_impl_original_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["LorBin", "bin", "-o", "out", "-fa", "input.fa", "-b", "input.bam"],
    )
    args = parser_args()
    assert args.recluster_impl == "original"


def test_bin_recluster_impl_cuda(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "LorBin",
            "bin",
            "-o",
            "out",
            "-fa",
            "input.fa",
            "-b",
            "input.bam",
            "--recluster_impl",
            "cuda",
        ],
    )
    args = parser_args()
    assert args.recluster_impl == "cuda"


def test_bin_max_cuda_points_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "LorBin",
            "bin",
            "-o",
            "out",
            "-fa",
            "input.fa",
            "-b",
            "input.bam",
            "--recluster_impl",
            "cuda",
            "--max_cuda_points",
            "20000",
        ],
    )
    args = parser_args()
    assert args.max_cuda_points == 20000


def test_bin_disable_cuda_fallback(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "LorBin",
            "bin",
            "-o",
            "out",
            "-fa",
            "input.fa",
            "-b",
            "input.bam",
            "--recluster_impl",
            "cuda",
            "--disable_cuda_fallback",
        ],
    )
    args = parser_args()
    assert args.disable_cuda_fallback is True
