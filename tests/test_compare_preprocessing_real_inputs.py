from lorbin.compare_preprocessing import run_compare


def test_run_compare_with_depth_bga(tmp_path):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c1\n" + ("ATGC" * 450) + "\n>c2\n" + ("TTAA" * 450) + "\n")

    depth = tmp_path / "depth.bga"
    depth.write_text(
        "c1\t0\t1000\t2\n"
        "c1\t1000\t1800\t4\n"
        "c2\t0\t900\t1\n"
        "c2\t900\t1800\t3\n"
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
