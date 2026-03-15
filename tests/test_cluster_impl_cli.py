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
