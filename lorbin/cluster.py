from collections import defaultdict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.neighbors import kneighbors_graph
try:
    from sklearn.neighbors import sort_graph_by_row_values
except ImportError:
    sort_graph_by_row_values = None
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from .utils import get_marker, process_fasta
from .model.KeepModel import KeepModel
from .model.EvaluationModel import EvaluationModel
import numpy as np
from sklearn.cluster import Birch, DBSCAN
from sklearn.exceptions import EfficiencyWarning
import logging
import os
import warnings

# Some sklearn versions may still emit sparse precomputed sorting warnings from internal checks.
warnings.filterwarnings(
    "ignore",
    category=EfficiencyWarning,
    message="Precomputed sparse input was not sorted by row values.*",
)


def _normalize_n_jobs(n_jobs):
    if n_jobs is None:
        return None
    if int(n_jobs) == 0:
        return -1
    return int(n_jobs)


def _effective_worker_count(n_jobs, task_count):
    normalized = _normalize_n_jobs(n_jobs)
    if task_count <= 1:
        return 1
    if normalized is None:
        return 1
    if normalized < 0:
        return max(1, min(task_count, os.cpu_count() or 1))
    return max(1, min(task_count, normalized))


def _counter_from_markers(markers):
    return Counter(markers) if markers else Counter()


def _build_candidate_cache(resultpool, contig_to_marker, namelist, contig_dict):
    contig_bp = [len(contig_dict[name]) for name in namelist]
    contig_marker_counts = [
        _counter_from_markers(contig_to_marker.get(name, ()))
        for name in namelist
    ]
    candidate_cache = []
    for bin_contig_index in resultpool:
        total_bp = 0
        marker_counts = Counter()
        for contig_index in bin_contig_index:
            total_bp += contig_bp[contig_index]
            marker_counts.update(contig_marker_counts[contig_index])
        candidate_cache.append(
            {
                "total_bp": total_bp,
                "marker_counts": marker_counts,
                "unique_marker_count": len(marker_counts),
                "total_marker_count": sum(marker_counts.values()),
            }
        )
    return candidate_cache, contig_bp, contig_marker_counts


def _marker_counter_to_sentence(marker_counts):
    parts = []
    for marker, count in marker_counts.items():
        parts.extend([marker] * count)
    return " ".join(parts)


def _update_candidate_stat_for_removed_contig(candidate_stat, contig_marker_count, contig_bp):
    candidate_stat["total_bp"] -= contig_bp
    if contig_marker_count:
        marker_counts = candidate_stat["marker_counts"]
        for marker, count in contig_marker_count.items():
            remaining = marker_counts.get(marker, 0) - count
            if remaining > 0:
                marker_counts[marker] = remaining
            elif marker in marker_counts:
                del marker_counts[marker]
        candidate_stat["unique_marker_count"] = len(marker_counts)
        candidate_stat["total_marker_count"] = sum(marker_counts.values())




def get_bin_best(keepmodel,evaluationmodel, resultpool, contig_to_marker, namelist, contig_dict, minfasta,a=0.6, candidate_cache=None):
    with torch.no_grad():
        for max_contamination in [0.1, 0.2, 0.3,0.4, 0.5,1]:
            max_F1 = 0
            weight_of_max = 1e9
            max_bin = None
            bin_recall = 0
            bin_contamination = 0

            for pool_index, bin_contig_index in enumerate(resultpool):
                stats = candidate_cache[pool_index] if candidate_cache is not None else None
                cur_weight = stats["total_bp"] if stats is not None else sum(len(contig_dict[namelist[contig_index]]) for contig_index in bin_contig_index)
                if cur_weight < minfasta:
                    continue
                if stats is not None:
                    unique_marker_count = stats["unique_marker_count"]
                    total_marker_count = stats["total_marker_count"]
                else:
                    marker_list = []
                    for contig_index in bin_contig_index:
                        marker_list.extend(contig_to_marker[namelist[contig_index]])
                    unique_marker_count = len(set(marker_list))
                    total_marker_count = len(marker_list)
                if total_marker_count == 0:
                    continue
                recall = unique_marker_count / 107
                contamination = (total_marker_count - unique_marker_count) / total_marker_count
                if contamination <= max_contamination:
                    F1 = (evaluationmodel(torch.Tensor([[recall, 1 - contamination, abs(recall - 1 + contamination)]])))
                    if F1 > max_F1:
                        max_F1 = F1
                        weight_of_max = cur_weight
                        max_bin = bin_contig_index
                        bin_recall = recall
                        bin_contamination = contamination
                    elif F1 == max_F1 and cur_weight <= weight_of_max:
                        weight_of_max = cur_weight
                        max_bin = bin_contig_index
                        bin_recall = recall
                        bin_contamination = contamination
            if max_F1 > 0:  # if there is a bin with F1 > 0
                keep = keepmodel(torch.Tensor([[bin_recall, 1 - bin_contamination, abs(bin_recall - 1 + bin_contamination)]]))
                keep = bool(keep.item() > a)
                return max_bin, keep

def get_bin_best_markers(keepmodel,evaluationmodel, vectorizer, tfidf_transformer, resultpool, contig_to_marker, namelist, contig_dict, minfasta,a=0.6, candidate_cache=None):
    with torch.no_grad():
        for max_contamination in [0.1, 0.2, 0.3,0.4, 0.5,1]:
            max_F1 = 0
            weight_of_max = 1e9
            max_bin = None
            bin_recall = 0
            bin_contamination = 0

            for pool_index, bin_contig_index in enumerate(resultpool):
                stats = candidate_cache[pool_index] if candidate_cache is not None else None
                cur_weight = stats["total_bp"] if stats is not None else sum(len(contig_dict[namelist[contig_index]]) for contig_index in bin_contig_index)
                if cur_weight < minfasta:
                    continue
                if stats is not None:
                    total_marker_count = stats["total_marker_count"]
                    unique_marker_count = stats["unique_marker_count"]
                    marker_sentence = _marker_counter_to_sentence(stats["marker_counts"])
                else:
                    marker_list = []
                    for contig_index in bin_contig_index:
                        marker_list.extend(contig_to_marker[namelist[contig_index]])
                    total_marker_count = len(marker_list)
                    unique_marker_count = len(set(marker_list))
                    marker_sentence = " ".join(marker_list)
                if total_marker_count == 0:
                    continue
                X = vectorizer.transform([marker_sentence])
                X = X.toarray()
                X_tfidf = tfidf_transformer.fit_transform(X)
                X = X_tfidf.toarray()

                recall = unique_marker_count / 107
                contamination = (total_marker_count - unique_marker_count) / total_marker_count
                if contamination <= max_contamination:
                    x_in1 = np.array([[recall, 1 - contamination, abs(recall - 1 + contamination)]])
                    x_in = np.concatenate((x_in1, X), axis=1)
                    x_in = torch.from_numpy(x_in).float()
                    F1 = evaluationmodel(x_in)
                    if F1 > max_F1:
                        max_F1 = F1
                        weight_of_max = cur_weight
                        max_bin = bin_contig_index
                        bin_recall = recall
                        bin_contamination = contamination
                        bin_x = X
                    elif F1 == max_F1 and cur_weight <= weight_of_max:
                        weight_of_max = cur_weight
                        max_bin = bin_contig_index
                        bin_recall = recall
                        bin_contamination = contamination
                        bin_x = X
            if max_F1 > 0:
                x_in1 = np.array([[bin_recall, 1 - bin_contamination, abs(bin_recall - 1 + bin_contamination)]])
                x_in = np.concatenate((x_in1, bin_x), axis=1)
                x_in = torch.from_numpy(x_in).float()
                keep = keepmodel(x_in)
                keep = bool(keep.item() > a)
                return max_bin, keep


def _prune_resultpool_original(resultpool, selected_contigs, candidate_cache=None, contig_bp=None, contig_marker_counts=None):
    for temp in selected_contigs:
        for pool_index, result in enumerate(resultpool):
            while temp in result:
                result.remove(temp)
                if candidate_cache is not None:
                    _update_candidate_stat_for_removed_contig(
                        candidate_cache[pool_index],
                        contig_marker_counts[temp],
                        contig_bp[temp],
                    )
    if candidate_cache is not None:
        for pool_index in range(len(candidate_cache) - 1, -1, -1):
            if not resultpool[pool_index]:
                del resultpool[pool_index]
                del candidate_cache[pool_index]


def _prune_resultpool_optimized(resultpool, selected_contigs, candidate_cache=None, contig_bp=None, contig_marker_counts=None):
    to_remove = set(selected_contigs)
    write_index = 0
    for read_index, result in enumerate(resultpool):
        if not result:
            continue
        pruned = []
        if candidate_cache is not None:
            candidate_stat = candidate_cache[read_index]
        else:
            candidate_stat = None
        for idx in result:
            if idx in to_remove:
                if candidate_stat is not None:
                    _update_candidate_stat_for_removed_contig(
                        candidate_stat,
                        contig_marker_counts[idx],
                        contig_bp[idx],
                    )
                continue
            pruned.append(idx)
        if pruned:
            resultpool[write_index] = pruned
            if candidate_cache is not None:
                candidate_cache[write_index] = candidate_stat
            write_index += 1
    del resultpool[write_index:]
    if candidate_cache is not None:
        del candidate_cache[write_index:]


def _get_total_bp(contig_names, contig_dict):
    return sum(len(contig_dict[name]) for name in contig_names)


def _connected_components_from_adjacency(adj):
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    components = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            neighbors = np.flatnonzero(adj[node])
            for nb in neighbors:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        components.append(comp)
    return components


def _normalize_thresholds(thresholds):
    normalized = sorted({float(t) for t in thresholds if float(t) >= 0.00001})
    return normalized


def _prepare_thresholds(thresholds, approximate_pruning=False, relative_tolerance=0.05):
    normalized = _normalize_thresholds(thresholds)
    if not approximate_pruning or not normalized:
        return normalized

    pruned = [normalized[0]]
    for threshold in normalized[1:]:
        prev = pruned[-1]
        denom = max(abs(prev), 1e-9)
        if abs(threshold - prev) / denom > relative_tolerance:
            pruned.append(threshold)
    return pruned


def _dedupe_resultpool(resultpool):
    if not resultpool:
        return []
    unique_resultpool = {tuple(sorted(comp)) for comp in resultpool if len(comp) > 1}
    return [list(comp) for comp in unique_resultpool]


def _labels_to_resultpool(labels, recluster_index):
    grouped = defaultdict(list)
    for label, name_index in zip(labels, recluster_index):
        if label != -1:
            grouped[int(label)].append(name_index)
    return list(grouped.values())


def _birch_result_for_threshold(threshold, recluster_latent, recluster_index):
    birch = Birch(threshold=threshold, n_clusters=None, branching_factor=50)
    labels = birch.fit_predict(recluster_latent)
    return _labels_to_resultpool(labels, recluster_index)


def _build_recluster_pool_birch_cpu(recluster_latent, recluster_index, thresholds, n_jobs=None, approximate_threshold_pruning=False):
    resultpool = []
    prepared_thresholds = _prepare_thresholds(
        thresholds,
        approximate_pruning=approximate_threshold_pruning,
    )
    max_workers = _effective_worker_count(n_jobs, len(prepared_thresholds))
    no_new_candidate_rounds = 0

    def extend_with_candidate_batches(candidate_batches):
        nonlocal no_new_candidate_rounds
        before = len(resultpool)
        for batch in candidate_batches:
            resultpool.extend(batch)
        resultpool[:] = _dedupe_resultpool(resultpool)
        if len(resultpool) == before:
            no_new_candidate_rounds += 1
        else:
            no_new_candidate_rounds = 0

    if max_workers == 1:
        for threshold in prepared_thresholds:
            extend_with_candidate_batches(
                [_birch_result_for_threshold(threshold, recluster_latent, recluster_index)]
            )
            if approximate_threshold_pruning and no_new_candidate_rounds >= 3:
                break
        return resultpool

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = []
        for threshold in prepared_thresholds:
            pending.append(executor.submit(_birch_result_for_threshold, threshold, recluster_latent, recluster_index))
            if len(pending) >= max_workers:
                extend_with_candidate_batches([future.result() for future in pending])
                pending = []
                if approximate_threshold_pruning and no_new_candidate_rounds >= 3:
                    break
        if pending:
            extend_with_candidate_batches([future.result() for future in pending])
    return _dedupe_resultpool(resultpool)


def _build_recluster_pool_birch_cpu_original(recluster_latent, recluster_index, thresholds):
    resultpool = []
    for threshold in thresholds:
        if threshold < 0.00001:
            continue
        birch = Birch(threshold=threshold, n_clusters=None, branching_factor=50)
        labels = birch.fit_predict(recluster_latent)
        grouped = defaultdict(list)
        for label, name_index in zip(labels, recluster_index):
            if label != -1:
                grouped[label].append(name_index)
        for res in grouped.values():
            resultpool.append(res)
    unique_resultpool = set(map(tuple, resultpool))
    return list(map(list, unique_resultpool))


def _estimate_auto_cuda_point_limit(recluster_latent, dtype_bytes, quadratic=False):
    n_dim = int(recluster_latent.shape[1]) if recluster_latent.ndim > 1 else 1
    if quadratic:
        # graph_cuda materializes pairwise distance and adjacency; reserve generous headroom.
        bytes_per_pair = 8.0
        reserve = 0.30
    else:
        # birch_cuda keeps one latent tensor on GPU plus CF stats, so memory scales ~linearly.
        bytes_per_pair = None
        reserve = 0.20

    try:
        total_vram = float(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return None

    budget = max(0.0, total_vram * (1.0 - reserve))
    if quadratic:
        if budget <= 0:
            return None
        return max(1, int((budget / bytes_per_pair) ** 0.5))

    bytes_per_point = max(1, n_dim) * float(dtype_bytes) * 3.0
    if budget <= 0:
        return None
    return max(1, int(budget / bytes_per_point))


def _resolve_cuda_point_limit(logger, recluster_impl, recluster_latent, max_cuda_points, dtype_bytes, quadratic=False):
    auto_limit = _estimate_auto_cuda_point_limit(recluster_latent, dtype_bytes=dtype_bytes, quadratic=quadratic)
    requested = int(max_cuda_points) if max_cuda_points is not None else 0
    if requested <= 0:
        if auto_limit is not None:
            logger.info(f"recluster_impl={recluster_impl} auto max_cuda_points={auto_limit} (derived from GPU VRAM)")
            return auto_limit
        logger.warning(f"recluster_impl={recluster_impl} could not auto-derive max_cuda_points; using conservative fallback=12000")
        return 12000

    if auto_limit is not None and requested > auto_limit:
        logger.warning(
            f"recluster_impl={recluster_impl} requested max_cuda_points={requested} exceeds VRAM-derived safe limit={auto_limit}; clamping to avoid OOM"
        )
        return auto_limit
    return requested


def _build_recluster_pool_graph_cuda(logger, recluster_latent, recluster_index, thresholds, max_cuda_points=0, cuda_fallback=True, effective_max_cuda_points=None):
    if not torch.cuda.is_available():
        msg = "recluster_impl=graph_cuda requested but CUDA is unavailable"
        if cuda_fallback:
            logger.warning(f"{msg}; falling back to optimized CPU recluster")
            return None
        raise RuntimeError(msg)

    if effective_max_cuda_points is None:
        effective_max_cuda_points = _resolve_cuda_point_limit(
            logger,
            "graph_cuda",
            recluster_latent,
            max_cuda_points,
            dtype_bytes=4,
            quadratic=True,
        )

    n_points = recluster_latent.shape[0]
    if n_points > effective_max_cuda_points:
        msg = f"recluster_impl=graph_cuda requested but {n_points} points exceed max_cuda_points={effective_max_cuda_points}"
        if cuda_fallback:
            logger.warning(f"{msg}; falling back to optimized CPU recluster")
            return None
        raise RuntimeError(msg)

    latent_tensor = torch.as_tensor(recluster_latent, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        dist = torch.cdist(latent_tensor, latent_tensor, p=2).detach().cpu().numpy()

    resultpool = []
    for threshold in _normalize_thresholds(thresholds):
        adjacency = dist <= threshold
        np.fill_diagonal(adjacency, True)
        components = _connected_components_from_adjacency(adjacency)
        for comp in components:
            if len(comp) > 1:
                resultpool.append([recluster_index[i] for i in comp])

    return _dedupe_resultpool(resultpool)


class _CFTreeGPU:
    def __init__(self, threshold, branching_factor=50, dtype=torch.float64):
        self.threshold = float(threshold)
        self.branching_factor = int(branching_factor)
        self.dtype = dtype
        self.counts = []
        self.sums = []
        self.sq_sums = []
        self.assignments = []

    def _device(self):
        if self.sums:
            return self.sums[0].device
        return torch.device("cuda")

    def _centroids(self):
        if not self.sums:
            return None
        counts = torch.tensor(self.counts, dtype=self.dtype, device=self._device()).unsqueeze(1)
        sums = torch.stack(self.sums)
        return sums / counts

    def find_best_subcluster(self, x):
        if not self.sums:
            return None
        centroids = self._centroids()
        dists = torch.linalg.norm(centroids - x.unsqueeze(0), dim=1)
        return int(torch.argmin(dists).item())

    def radius_after_merge(self, subcluster_index, x):
        if subcluster_index is None:
            return 0.0
        old_count = self.counts[subcluster_index]
        new_count = old_count + 1
        new_sum = self.sums[subcluster_index] + x
        new_sq_sum = self.sq_sums[subcluster_index] + torch.dot(x, x)
        centroid = new_sum / new_count
        mean_sq_norm = new_sq_sum / new_count
        radius_sq = mean_sq_norm - torch.dot(centroid, centroid)
        return float(torch.sqrt(torch.clamp(radius_sq, min=0.0)).item())

    def merge(self, subcluster_index, x):
        self.counts[subcluster_index] += 1
        self.sums[subcluster_index] = self.sums[subcluster_index] + x
        self.sq_sums[subcluster_index] = self.sq_sums[subcluster_index] + torch.dot(x, x)
        self.assignments.append(subcluster_index)

    def add_new_subcluster(self, x):
        self.counts.append(1)
        self.sums.append(x.clone())
        self.sq_sums.append(torch.dot(x, x))
        self.assignments.append(len(self.sums) - 1)

    def split_upward_if_needed(self):
        # Compact leaf-level subclusters when we exceed branching factor.
        if len(self.sums) <= self.branching_factor:
            return
        centroids = self._centroids()
        # Merge the closest two subclusters; this keeps the structure bounded and close to BIRCH CF compaction.
        dmat = torch.cdist(centroids, centroids)
        inf = torch.tensor(float("inf"), device=dmat.device, dtype=dmat.dtype)
        dmat.fill_diagonal_(inf)
        flat_idx = int(torch.argmin(dmat).item())
        n = dmat.shape[0]
        i = flat_idx // n
        j = flat_idx % n
        if i > j:
            i, j = j, i
        self.counts[i] += self.counts[j]
        self.sums[i] = self.sums[i] + self.sums[j]
        self.sq_sums[i] = self.sq_sums[i] + self.sq_sums[j]
        del self.counts[j]
        del self.sums[j]
        del self.sq_sums[j]
        # Re-map historical assignments to preserve insertion order labels.
        remapped = []
        for a in self.assignments:
            if a == j:
                remapped.append(i)
            elif a > j:
                remapped.append(a - 1)
            else:
                remapped.append(a)
        self.assignments = remapped

    def emit_leaf_labels(self):
        return np.array(self.assignments, dtype=int)


def _build_recluster_pool_birch_cuda(logger, recluster_latent, recluster_index, thresholds, max_cuda_points=0, cuda_fallback=True, effective_max_cuda_points=None):
    if not torch.cuda.is_available():
        msg = "recluster_impl=birch_cuda requested but CUDA is unavailable"
        if cuda_fallback:
            logger.warning(f"{msg}; falling back to optimized CPU recluster")
            return None
        raise RuntimeError(msg)

    if effective_max_cuda_points is None:
        effective_max_cuda_points = _resolve_cuda_point_limit(
            logger,
            "birch_cuda",
            recluster_latent,
            max_cuda_points,
            dtype_bytes=8,
            quadratic=False,
        )

    n_points = recluster_latent.shape[0]
    if n_points > effective_max_cuda_points:
        msg = f"recluster_impl=birch_cuda requested but {n_points} points exceed max_cuda_points={effective_max_cuda_points}"
        if cuda_fallback:
            logger.warning(f"{msg}; falling back to optimized CPU recluster")
            return None
        raise RuntimeError(msg)

    # Correctness-first mode: use the exact sklearn BIRCH candidate generation that
    # the original stage-2 path uses, while keeping CUDA availability/limit checks.
    # The previous custom CFTreeGPU path was not algorithmically equivalent and
    # could under-generate candidate bins.
    logger.info("recluster_impl=birch_cuda using sklearn BIRCH-compatible candidate generation for stage-2 equivalence")
    return _build_recluster_pool_birch_cpu_original(recluster_latent, recluster_index, thresholds)


def _build_recluster_pool_cuda(logger, recluster_latent, recluster_index, thresholds, max_cuda_points=0, cuda_fallback=True, effective_max_cuda_points=None):
    # Backward-compatible alias for CUDA recluster implementation.
    return _build_recluster_pool_birch_cuda(
        logger,
        recluster_latent,
        recluster_index,
        thresholds,
        max_cuda_points=max_cuda_points,
        cuda_fallback=cuda_fallback,
        effective_max_cuda_points=effective_max_cuda_points,
    )

def bin_cluster(logger, latent, contig2marker, contig_dict, contig_list, contig_all, minfasta, feature="no_markers", a=0.6, cluster_impl="optimized", recluster_impl="original", max_cuda_points=0, cuda_fallback=True, cluster_n_jobs=None, approximate_threshold_pruning=False):
    # Defensive normalization keeps runtime robust even under partially updated installs.
    if max_cuda_points is None:
        max_cuda_points = 0
    if cuda_fallback is None:
        cuda_fallback = True

    use_optimized = cluster_impl == "optimized"
    use_recluster_optimized = recluster_impl in {"optimized", "cuda", "birch_cuda", "graph_cuda"}

    result_dict = {}
    # create BIRCH
    min_k_1 = min(200, latent.shape[0] - 1)
    dist_matrix = kneighbors_graph(
        latent,
        n_neighbors=min_k_1,
        mode='distance',
        p=2,
        n_jobs=_normalize_n_jobs(cluster_n_jobs))
    dist_matrix_cos = kneighbors_graph(
        latent,
        n_neighbors=min_k_1,
        mode='distance',
        metric="cosine",
        p=2,
        n_jobs=_normalize_n_jobs(cluster_n_jobs))

    if sort_graph_by_row_values is not None:
        dist_matrix = sort_graph_by_row_values(dist_matrix, warn_when_not_sorted=False)
        dist_matrix_cos = sort_graph_by_row_values(dist_matrix_cos, warn_when_not_sorted=False)

    cos_distance_data = dist_matrix_cos.data
    p2_distance_data = dist_matrix.data

    length_weight = np.array(
        [len(contig_dict[name]) for name in contig_list])
    resultpool = []
    contig_list_index = [i for i in range(len(contig_list))]

    eps_cos=[]
    eps_p2=[]
    for k in range(1,min(min_k_1,80)):
        eps_value=np.partition(cos_distance_data,latent.shape[0]*k-1)[latent.shape[0]*k-1]
        eps_cos.append(round(0.7*eps_value,2))
        eps_cos.append(round(1.2 * eps_value,2))
    eps_cos = list(set(eps_cos))
    eps_cos.sort()
    eps_cos = [ i for i in eps_cos if i<0.1] +[0.12,0.15,0.2]
    for k in range(1,min(80,min_k_1)):
        eps_value=np.partition(p2_distance_data,latent.shape[0]*k-1)[latent.shape[0]*k-1]
        eps_p2.append(math.floor(0.5*eps_value))
        eps_p2.append(math.floor(eps_value))
        eps_p2.append(math.floor(1.2 * eps_value))

    eps_p2 = list(set(eps_p2))
    eps_p2.sort()
    eps_p2 = eps_p2+[0.3*eps_p2[0],0.5*eps_p2[0],eps_p2[0]+0.5,eps_p2[1]+0.5]
    eps_p2.sort()


    for e,eps_value in enumerate(eps_p2):
        if eps_value<0.001:
            continue
        if e<5:
            minsample=5
        else:
            minsample=5
        dbscan = DBSCAN(eps=eps_value, min_samples=minsample, n_jobs=_normalize_n_jobs(cluster_n_jobs), metric='precomputed')
        dbscan.fit(dist_matrix, sample_weight=length_weight)
        # predict the labal
        labels = dbscan.labels_
        res_temp = defaultdict(list)
        for label, name in zip(labels, contig_list_index):
            if label != -1:
                res_temp[label].append(name)
        resultpool.extend(res_temp.values())


    for e,eps_value in enumerate(eps_cos):
        if eps_value<0.001:
            continue
        if e<5:
            minsample=5
        else:
            minsample=5
        dbscan = DBSCAN(eps=eps_value, min_samples=minsample, n_jobs=_normalize_n_jobs(cluster_n_jobs), metric='precomputed')
        dbscan.fit(dist_matrix_cos, sample_weight=length_weight)
        labels = dbscan.labels_
        result_dict[eps_value] = labels.tolist()

        res_temp = defaultdict(list)
        for label, name in zip(labels, contig_list_index):
            if label != -1:
                res_temp[f"cos_{label}"].append(name)
        resultpool.extend(res_temp.values())
    unique_resultpool = set(map(tuple, resultpool))
    resultpool = list(map(list, unique_resultpool))
    modeldir = os.path.split(__file__)[0]
    extracted = []
    all_extracted=[]
    if feature=="no_markers":
        model1 = EvaluationModel(3)
        model1.load_state_dict(torch.load(f'{modeldir}/model/huigui_weights.pt'))
        model2 = KeepModel(3)
        model2.load_state_dict(torch.load(f'{modeldir}/model/keepmodel_weights.pt'))
    elif feature=="markers110":
        model1 = EvaluationModel(110)
        model1.load_state_dict(torch.load(f'{modeldir}/model/huigui_weights_markers110.pt'))
        model2 = KeepModel(110)
        model2.load_state_dict(torch.load(f'{modeldir}/model/keepmodel_weights_markers110.pt'))
        vocabulary_df = pd.read_csv(f'{modeldir}/model/single-copy-genes.csv', index_col=0)
        vocabulary = list(vocabulary_df.iloc[:, 0])
        vectorizer = CountVectorizer(vocabulary=vocabulary, lowercase=False)
        tfidf_transformer = TfidfTransformer()
    else:
        model1 = EvaluationModel(35)
        model1.load_state_dict(torch.load(f'{modeldir}/model/huigui_weights_markers35.pt'))
        model2 = KeepModel(35)
        model2.load_state_dict(torch.load(f'{modeldir}/model/keepmodel_weights_markers35.pt'))
        vocabulary_df = pd.read_csv(f'{modeldir}/model/single-copy-genes35.csv', index_col=0)
        vocabulary = list(vocabulary_df.iloc[:, 0])
        vectorizer = CountVectorizer(vocabulary=vocabulary, lowercase=False)
        tfidf_transformer = TfidfTransformer()
    logger.info("load cluster quality asssement model and clustering decision model")
    keep_label=[]
    keep_count=0
    logger.info("cluster")
    remaining_bp = _get_total_bp(contig_list, contig_dict)
    candidate_cache, contig_bp, contig_marker_counts = _build_candidate_cache(
        resultpool,
        contig2marker,
        contig_all,
        contig_dict,
    )

    while remaining_bp >= minfasta:
        if len(contig_list) == 1:
            extracted.append(contig_list_index)
            break
        if feature=="no_markers":
            max_bin = get_bin_best(model2, model1, resultpool, contig2marker, contig_all, contig_dict, minfasta,a, candidate_cache=candidate_cache)
        else:
            max_bin = get_bin_best_markers(model2, model1, vectorizer, tfidf_transformer,resultpool, contig2marker, contig_all, contig_dict, minfasta,a, candidate_cache=candidate_cache)
        if not max_bin or keep_count>200:
            break
        max_bin, keep = max_bin
        if keep:
            keep_count=0
            extracted.append(max_bin.copy())
        else:
            keep_count+=1
        all_extracted.append(max_bin.copy())
        keep_label.append(keep)
        remaining_bp -= sum(contig_bp[idx] for idx in max_bin)
        if use_optimized:
            _prune_resultpool_optimized(resultpool, max_bin.copy(), candidate_cache=candidate_cache, contig_bp=contig_bp, contig_marker_counts=contig_marker_counts)
        else:
            _prune_resultpool_original(resultpool, max_bin.copy(), candidate_cache=candidate_cache, contig_bp=contig_bp, contig_marker_counts=contig_marker_counts)

    contig2ix = {}
    label_index=0
    for i, cs in enumerate(extracted):
        for c in cs:
            contig2ix[contig_all[c]] = i
        label_index=i+1
    contig_labels = [contig2ix.get(c, -1) for c in contig_all]
    contig2ix_={}
    repeat=[]
    for i, cs in enumerate(all_extracted):
        for c in cs:
            if contig2ix_.get(contig_all[c], -1)==-1:
                contig2ix_[contig_all[c]] = i
            else:
                repeat.append(c)
        label_index=i+1
    contig_labels_ = [contig2ix_.get(c, -1) for c in contig_all]
    logger.info('start recluster')
    logger.info(f"recluster config: impl={recluster_impl}, max_cuda_points={max_cuda_points}, cuda_fallback={cuda_fallback}")
    recluster_index = [index for index, value in enumerate(contig_labels) if value == -1]
    if len(recluster_index) <= 1:
        return contig_labels,keep_label

    recluster_latent = latent[recluster_index]
    recluster_list = contig_all[recluster_index].tolist()
    min_k_2 = min(200, recluster_latent.shape[0]-1)
    if min_k_2 < 5:
        return contig_labels,keep_label

    dist_matrix = kneighbors_graph(
        recluster_latent,
        n_neighbors=min_k_2,
        mode='distance',
        p=2,
        n_jobs=_normalize_n_jobs(cluster_n_jobs))
    if sort_graph_by_row_values is not None:
        dist_matrix = sort_graph_by_row_values(dist_matrix, warn_when_not_sorted=False)

    p2_distance = dist_matrix.data
    eps_p2_2=[]


    resultpool = []
    for k in range(5,min(80,min_k_2),10):
        eps_value=np.partition(p2_distance,recluster_latent.shape[0]*k-1)[recluster_latent.shape[0]*k-1]
        eps_p2_2.append(math.floor(0.8*eps_value))
        eps_p2_2.append(math.floor(eps_value))
        eps_p2_2.append(math.floor(1.2*eps_value))
    eps_p2_2 = list(set(eps_p2_2))
    eps_p2_2.sort()

    if not eps_p2_2:
        return contig_labels,keep_label

    threds =eps_p2_2 +[0.1*eps_p2_2[0],0.3*eps_p2_2[0],0.5*eps_p2_2[0],10,15,30]
    threds.sort()

    if recluster_impl == "cuda":
        logger.warning("recluster_impl=cuda is a legacy alias that now uses the optimized CPU BIRCH path; use birch_cuda or graph_cuda for CUDA-gated reclustering")
        recluster_impl = "optimized"

    if recluster_impl == "graph_cuda":
        effective_max_cuda_points = _resolve_cuda_point_limit(
            logger,
            "graph_cuda",
            recluster_latent,
            max_cuda_points,
            dtype_bytes=4,
            quadratic=True,
        )
        logger.info(
            f"stage2_points={len(recluster_index)} effective_max_cuda_points={effective_max_cuda_points} builder=graph_cuda"
        )
        cuda_resultpool = _build_recluster_pool_graph_cuda(
            logger,
            recluster_latent,
            recluster_index,
            threds,
            max_cuda_points=max_cuda_points,
            cuda_fallback=cuda_fallback,
            effective_max_cuda_points=effective_max_cuda_points,
        )
        if cuda_resultpool is None:
            recluster_impl = "optimized"
        else:
            resultpool = cuda_resultpool
    elif recluster_impl == "birch_cuda":
        effective_max_cuda_points = _resolve_cuda_point_limit(
            logger,
            "birch_cuda",
            recluster_latent,
            max_cuda_points,
            dtype_bytes=8,
            quadratic=False,
        )
        logger.info(
            f"stage2_points={len(recluster_index)} effective_max_cuda_points={effective_max_cuda_points} builder=birch_cuda"
        )
        cuda_resultpool = _build_recluster_pool_birch_cuda(
            logger,
            recluster_latent,
            recluster_index,
            threds,
            max_cuda_points=max_cuda_points,
            cuda_fallback=cuda_fallback,
            effective_max_cuda_points=effective_max_cuda_points,
        )
        if cuda_resultpool is None:
            recluster_impl = "optimized"
        else:
            resultpool = cuda_resultpool

    if recluster_impl == "original":
        logger.info(
            f"stage2_points={len(recluster_index)} effective_max_cuda_points=n/a builder=original"
        )
        resultpool = _build_recluster_pool_birch_cpu_original(recluster_latent, recluster_index, threds)
    elif recluster_impl == "optimized":
        logger.info(
            f"stage2_points={len(recluster_index)} effective_max_cuda_points=n/a builder=optimized"
        )
        resultpool = _build_recluster_pool_birch_cpu(
            recluster_latent,
            recluster_index,
            threds,
            n_jobs=cluster_n_jobs,
            approximate_threshold_pruning=approximate_threshold_pruning,
        )
    if not resultpool:
        return contig_labels,keep_label

    minfasta = 1500
    recluster_remaining_bp = _get_total_bp(recluster_list, contig_dict)
    candidate_cache, _, _ = _build_candidate_cache(
        resultpool,
        contig2marker,
        contig_all,
        contig_dict,
    )
    while recluster_remaining_bp >= minfasta:
        if len(recluster_list) == 1:
            extracted.append(recluster_index)
            break

        if feature=="no_markers":
            max_bin = get_bin_best(model2, model1, resultpool, contig2marker, contig_all, contig_dict, minfasta,a, candidate_cache=candidate_cache)
        else:
            max_bin = get_bin_best_markers(model2, model1, vectorizer, tfidf_transformer,resultpool, contig2marker, contig_all, contig_dict, minfasta,a, candidate_cache=candidate_cache)
        if not max_bin:
            break
        max_bin, keep = max_bin

        extracted.append(max_bin.copy())
        recluster_remaining_bp -= sum(contig_bp[idx] for idx in max_bin)
        if use_recluster_optimized:
            _prune_resultpool_optimized(resultpool, max_bin.copy(), candidate_cache=candidate_cache, contig_bp=contig_bp, contig_marker_counts=contig_marker_counts)
        else:
            _prune_resultpool_original(resultpool, max_bin.copy(), candidate_cache=candidate_cache, contig_bp=contig_bp, contig_marker_counts=contig_marker_counts)
    contig2ix = {}
    for i, cs in enumerate(extracted):
        for c in cs:
            contig2ix[contig_all[c]] = i

    contig_labels = [contig2ix.get(c, -1) for c in contig_all]
    return contig_labels,keep_label
