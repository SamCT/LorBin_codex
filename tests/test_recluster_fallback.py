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
        recluster_impl="birch_cuda",
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
            recluster_impl="birch_cuda",
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
        recluster_impl="birch_cuda",
        max_cuda_points=None,
        cuda_fallback=None,
    )

    assert len(labels) == len(contig_all)
    assert isinstance(keep, list)


def test_cuda_alias_routes_to_optimized(monkeypatch):
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

    def fake_get_bin_best(_keepmodel, _evaluationmodel, resultpool, *_args, **_kwargs):
        if not resultpool:
            return None
        return resultpool[0], True

    monkeypatch.setattr(cluster_mod, "get_bin_best", fake_get_bin_best)

    calls = {"optimized": 0}

    def fake_optimized(_latent, _idx, _thresholds, **_kwargs):
        calls["optimized"] += 1
        return [[0, 1]]

    monkeypatch.setattr(cluster_mod, "_build_recluster_pool_birch_cpu", fake_optimized)

    class DummyLogger:
        def __init__(self):
            self.warnings = []

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, message, *_args, **_kwargs):
            self.warnings.append(message)

    logger = DummyLogger()
    labels, keep = cluster_mod.bin_cluster(
        logger,
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

    assert calls["optimized"] == 1
    assert any("legacy alias" in warning for warning in logger.warnings)
    assert len(labels) == len(contig_all)
    assert isinstance(keep, list)


def test_recluster_not_limited_to_100_bins(monkeypatch):
    n = 150
    latent = np.array([[float(i), 0.0] for i in range(n)], dtype=np.float32)
    contig_all = np.array([f"c{i}" for i in range(n)])
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
            # one singleton cluster per point, creates >100 stage-2 candidates
            return np.arange(len(X), dtype=int)

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

    labels, _keep = cluster_mod.bin_cluster(
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
        recluster_impl="original",
    )

    # Regression: stage-2 should not stop at 100 iterations.
    assert len(set(labels)) > 100


def test_birch_cuda_uses_cpu_original_candidate_builder(monkeypatch):
    latent = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    recluster_index = [3, 7]
    thresholds = [0.5, 1.0]

    monkeypatch.setattr(cluster_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cluster_mod, "_resolve_cuda_point_limit", lambda *a, **k: 1000)

    captured = {}

    def fake_cpu_original(latent_in, idx_in, threds_in):
        captured["latent_shape"] = latent_in.shape
        captured["idx"] = idx_in
        captured["threds"] = threds_in
        return [[idx_in[0]], [idx_in[1]]]

    monkeypatch.setattr(cluster_mod, "_build_recluster_pool_birch_cpu_original", fake_cpu_original)

    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    out = cluster_mod._build_recluster_pool_birch_cuda(
        DummyLogger(),
        latent,
        recluster_index,
        thresholds,
        max_cuda_points=0,
        cuda_fallback=True,
    )

    assert out == [[3], [7]]
    assert captured["latent_shape"] == latent.shape
    assert captured["idx"] == recluster_index
    assert captured["threds"] == thresholds


def test_optimized_2_recluster_honors_keep_and_continues(monkeypatch):
    latent = np.array([[float(i), 0.0] for i in range(6)], dtype=np.float32)
    contig_all = np.array([f"c{i}" for i in range(6)])
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
    monkeypatch.setattr(cluster_mod, "_compute_stage2_thresholds_optimized_2", lambda _dist: [0.25, 0.5])
    monkeypatch.setattr(cluster_mod, "_build_recluster_pool_birch_cpu", lambda *_a, **_k: [[0, 1, 2], [3, 4, 5]])

    selections = iter([([0, 1, 2], False), ([3, 4, 5], True)])
    calls = {"count": 0}

    def fake_get_bin_best(_keepmodel, _evaluationmodel, resultpool, *_args, **_kwargs):
        assert all(isinstance(pool, set) for pool in resultpool)
        calls["count"] += 1
        try:
            return next(selections)
        except StopIteration:
            return None

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
        recluster_impl="optimized_2",
    )

    assert calls["count"] == 2
    assert labels[:3] == [-1, -1, -1]
    assert labels[3:] == [0, 0, 0]
    assert isinstance(keep, list)
