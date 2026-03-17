import numpy as np

from lorbin import cluster as cluster_mod


def test_cuda_fallback_populates_resultpool(monkeypatch):
    latent = np.array([[float(i), 0.0] for i in range(12)], dtype=np.float32)
    contig_all = np.array([f"c{i}" for i in range(12)])
    contig_list = contig_all.tolist()
    contig_dict = {name: "A" * 2000 for name in contig_list}
    contig2marker = {name: ["m1"] for name in contig_list}

    class DummyModel:
        def load_state_dict(self, _):
            return None

        def __call__(self, _):
            return 1.0

    monkeypatch.setattr(cluster_mod, "EvaluationModel", lambda *_: DummyModel())
    monkeypatch.setattr(cluster_mod, "KeepModel", lambda *_: DummyModel())
    monkeypatch.setattr(cluster_mod.torch, "load", lambda *_args, **_kwargs: {})

    class DummyBirch:
        def __init__(self, threshold, n_clusters=None, branching_factor=50, **_kwargs):
            self.threshold = threshold

        def fit_predict(self, X):
            n = len(X)
            labels = np.full(n, -1, dtype=int)
            labels[: max(2, n // 2)] = 0
            return labels

    monkeypatch.setattr(cluster_mod, "Birch", DummyBirch)

    # Produce no bins in stage 1 so all contigs go to recluster stage.
    monkeypatch.setattr(cluster_mod, "DBSCAN", lambda *a, **k: type("D", (), {"fit": lambda self, *_a, **_k: self, "labels_": np.full(len(latent), -1, dtype=int)})())

    def fake_get_bin_best(_keepmodel, _evaluationmodel, resultpool, *_args, **_kwargs):
        if not resultpool:
            return None
        return resultpool[0], True

    monkeypatch.setattr(cluster_mod, "get_bin_best", fake_get_bin_best)

    # Simulate CUDA path returning None to force optimized fallback.
    monkeypatch.setattr(cluster_mod, "_build_recluster_pool_birch_cuda", lambda *_args, **_kwargs: None)

    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    labels, keep = cluster_mod.bin_cluster(
        DummyLogger(),
        latent,
        contig2marker,
        contig_dict,
        contig_list,
        contig_all,
        minfasta=10_000_000,
        feature="no_markers",
        a=0.6,
        cluster_impl="optimized",
        recluster_impl="cuda",
    )

    # Regression: should not crash and should produce labels via CPU fallback path.
    assert len(labels) == len(contig_all)
    assert isinstance(keep, list)


def test_cuda_no_fallback_raises(monkeypatch):
    latent = np.array([[float(i), 0.0] for i in range(12)], dtype=np.float32)
    contig_all = np.array([f"c{i}" for i in range(12)])
    contig_list = contig_all.tolist()
    contig_dict = {name: "A" * 2000 for name in contig_list}
    contig2marker = {name: ["m1"] for name in contig_list}

    class DummyModel:
        def load_state_dict(self, _):
            return None

        def __call__(self, _):
            return 1.0

    monkeypatch.setattr(cluster_mod, "EvaluationModel", lambda *_: DummyModel())
    monkeypatch.setattr(cluster_mod, "KeepModel", lambda *_: DummyModel())
    monkeypatch.setattr(cluster_mod.torch, "load", lambda *_args, **_kwargs: {})

    monkeypatch.setattr(cluster_mod, "DBSCAN", lambda *a, **k: type("D", (), {"fit": lambda self, *_a, **_k: self, "labels_": np.full(len(latent), -1, dtype=int)})())
    monkeypatch.setattr(cluster_mod, "get_bin_best", lambda *_a, **_k: None)
    monkeypatch.setattr(cluster_mod, "_build_recluster_pool_birch_cuda", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("cuda unavailable")))

    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    try:
        cluster_mod.bin_cluster(
            DummyLogger(),
            latent,
            contig2marker,
            contig_dict,
            contig_list,
            contig_all,
            minfasta=10_000_000,
            feature="no_markers",
            a=0.6,
            cluster_impl="optimized",
            recluster_impl="cuda",
            cuda_fallback=False,
        )
    except RuntimeError as exc:
        assert "cuda" in str(exc).lower()
    else:
        raise AssertionError("Expected RuntimeError when cuda_fallback is disabled")


def test_recluster_none_config_values_are_normalized(monkeypatch):
    latent = np.array([[float(i), 0.0] for i in range(12)], dtype=np.float32)
    contig_all = np.array([f"c{i}" for i in range(12)])
    contig_list = contig_all.tolist()
    contig_dict = {name: "A" * 2000 for name in contig_list}
    contig2marker = {name: ["m1"] for name in contig_list}

    class DummyModel:
        def load_state_dict(self, _):
            return None

        def __call__(self, _):
            return 1.0

    monkeypatch.setattr(cluster_mod, "EvaluationModel", lambda *_: DummyModel())
    monkeypatch.setattr(cluster_mod, "KeepModel", lambda *_: DummyModel())
    monkeypatch.setattr(cluster_mod.torch, "load", lambda *_args, **_kwargs: {})

    class DummyBirch:
        def __init__(self, threshold, n_clusters=None, branching_factor=50, **_kwargs):
            self.threshold = threshold

        def fit_predict(self, X):
            n = len(X)
            labels = np.full(n, -1, dtype=int)
            labels[: max(2, n // 2)] = 0
            return labels

    monkeypatch.setattr(cluster_mod, "Birch", DummyBirch)
    monkeypatch.setattr(cluster_mod, "DBSCAN", lambda *a, **k: type("D", (), {"fit": lambda self, *_a, **_k: self, "labels_": np.full(len(latent), -1, dtype=int)})())

    def fake_get_bin_best(_keepmodel, _evaluationmodel, resultpool, *_args, **_kwargs):
        if not resultpool:
            return None
        return resultpool[0], True

    monkeypatch.setattr(cluster_mod, "get_bin_best", fake_get_bin_best)

    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    labels, keep = cluster_mod.bin_cluster(
        DummyLogger(),
        latent,
        contig2marker,
        contig_dict,
        contig_list,
        contig_all,
        minfasta=10_000_000,
        feature="no_markers",
        a=0.6,
        cluster_impl="optimized",
        recluster_impl="cuda",
        max_cuda_points=None,
        cuda_fallback=None,
    )

    assert len(labels) == len(contig_all)
    assert isinstance(keep, list)
