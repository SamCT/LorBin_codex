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
        method="both",
    )

    assert (outdir / "comparison_summary.json").exists()
    assert (outdir / "kmer_original.csv").exists()
    assert (outdir / "kmer_optimized.csv").exists()
    assert (outdir / "coverage_original.csv").exists()
    assert (outdir / "coverage_optimized.csv").exists()

    statuses = {item["name"]: item["exact_equal"] for item in summary["comparisons"]}
    assert statuses["kmer"] is True
    assert statuses["length"] is True
    assert statuses["coverage"] is True


def test_run_original_only(tmp_path):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c1\nATGCGATGCGAT\n")
    depth = tmp_path / "depth.bga"
    depth.write_text("c1\t0\t200\t3\n")

    outdir = tmp_path / "original_out"
    summary = run_compare(fasta=str(fasta), output_dir=str(outdir), depth_bga=str(depth), method="original")

    assert (outdir / "kmer_original.csv").exists()
    assert (outdir / "coverage_original.csv").exists()
    assert not (outdir / "kmer_optimized.csv").exists()
    assert summary["comparisons"] == []
    assert summary["parameters"]["method"] == "original"
