from pathlib import Path

from lorbin.compare_preprocessing import run_compare


def test_run_compare_with_depth_bga(tmp_path):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c1\nATGCGATGCGAT\n>c2\nTTTTAAAACCCC\n")

    depth = tmp_path / "depth.bga"
    depth.write_text(
        "c1\t0\t100\t2\n"
        "c1\t100\t200\t4\n"
        "c2\t0\t80\t1\n"
        "c2\t80\t180\t3\n"
    )

    outdir = tmp_path / "compare_out"
    summary = run_compare(
        fasta=str(fasta),
        output_dir=str(outdir),
        depth_bga=str(depth),
    )

    assert (outdir / "comparison_summary.json").exists()
    assert (outdir / "kmer_old.csv").exists()
    assert (outdir / "kmer_new.csv").exists()
    assert (outdir / "coverage_old.csv").exists()
    assert (outdir / "coverage_new.csv").exists()

    statuses = {item["name"]: item["exact_equal"] for item in summary["comparisons"]}
    assert statuses["kmer"] is True
    assert statuses["length"] is True
    assert statuses["coverage"] is True

    assert summary["parameters"]["edge"] == 75
    assert summary["parameters"]["contig_threshold"] == 1000
    assert summary["parameters"]["length_threshold"] == 1500
    assert summary["parameters"]["kmer_len"] == 4
