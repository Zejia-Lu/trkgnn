"""
Test: 验证重构后的 cluster 算法与原始实现结果等价。
运行方式:
    conda run -n deeplearning python jobs/test_cluster_equivalence.py
"""
import random
import numpy as np
import torch
import torch_geometric
import torch_geometric.utils as pyg_utils
from collections import defaultdict


# ──────────────────────────────────────────────
# 原始 Dijkstra 实现（直接从重构前代码复制）
# ──────────────────────────────────────────────
def cluster_original(gr, node_xyz, threshold=0.5, first_threshold=0.5, layer_centers=None, tolerance=1.0):
    differences = np.abs(gr.x[:, 2].numpy().reshape(-1, 1) - np.array(layer_centers).reshape(1, -1))
    layer_indices = np.argmin(differences, axis=1)
    layers_arr = np.where(np.min(differences, axis=1) <= tolerance, layer_indices, -1)
    layer_tensor = torch.tensor(layers_arr)
    gr.x = torch.cat([gr.x, layer_tensor.unsqueeze(-1)], dim=-1)

    mask = torch.where(
        gr.x[gr.edge_index[0], 2] < 9,
        gr.edge_attr[:, -2] > first_threshold,
        gr.edge_attr[:, -2] > threshold,
    )
    sub_data = torch_geometric.data.Data(
        x=gr.x,
        edge_index=gr.edge_index[:, mask],
        edge_attr=gr.edge_attr[mask, :],
    )

    out_degrees = pyg_utils.degree(sub_data.edge_index[0], sub_data.num_nodes)
    in_degrees  = pyg_utils.degree(sub_data.edge_index[1], sub_data.num_nodes)
    start_nodes = ((out_degrees > 0) & (in_degrees == 0)).nonzero(as_tuple=False).view(-1).numpy()

    import heapq

    edge_dict = defaultdict(dict)
    edge_index = sub_data.edge_index.cpu().numpy()
    edge_attr  = sub_data.edge_attr.cpu().numpy()
    for i in range(edge_index.shape[1]):
        s, e = int(edge_index[0][i]), int(edge_index[1][i])
        edge_dict[s][e] = float(edge_attr[i][-2])

    children = defaultdict(list)
    for s, e in edge_index.T:
        children[s].append(e)

    heap = []
    best_scores = defaultdict(lambda: -float('inf'))
    best_paths  = defaultdict(dict)

    for start_node in start_nodes:
        start_node = int(start_node)
        path = [start_node]
        heapq.heappush(heap, (0.0, start_node, path))
        best_scores[start_node] = 0.0
        best_paths[start_node][start_node] = (path, 0.0)

    layers_long = sub_data.x[:, -1].long()

    while heap:
        neg_score, current_node, path = heapq.heappop(heap)
        current_score = -neg_score
        if current_score < best_scores[current_node]:
            continue
        current_layer = layers_long[current_node].item()
        for child in children[current_node]:
            child_layer = layers_long[child].item()
            if child_layer != current_layer + 1:
                continue
            edge_score = edge_dict[current_node].get(child, 0.0)
            new_score  = current_score + edge_score
            new_path   = path + [child]
            start_node = path[0]
            if new_score > best_scores[child]:
                best_scores[child] = new_score
                best_paths[start_node][child] = (new_path, new_score)
                heapq.heappush(heap, (-new_score, child, new_path))

    def calc_avg(p):
        if len(p) <= 3:
            return 0.0
        total = sum(edge_dict[p[i]].get(p[i+1], 0.0) for i in range(len(p)-1))
        return total / (len(p) - 1)

    final_paths = defaultdict(list)
    node_to_best = {}
    used_nodes   = set()

    for sn in best_paths:
        for node in best_paths[sn]:
            path, score = best_paths[sn][node]
            int_path = [int(n) for n in path]
            avg = calc_avg(int_path)
            for n in path:
                coords = tuple(np.round(node_xyz[n, :3], decimals=4))
                if coords in node_to_best or coords in used_nodes:
                    if score > node_to_best[coords][1]:
                        node_to_best[coords] = (int_path, score, avg)
                else:
                    node_to_best[coords] = (int_path, score, avg)
                    used_nodes.add(coords)

    for coords, (path, score, avg) in node_to_best.items():
        sn = path[0]
        if path not in final_paths[sn]:
            final_paths[sn].append((path, avg))

    # clean shared hits
    all_paths = []
    for sn, pl in final_paths.items():
        for path, avg in pl:
            if len(path) > 3:
                all_paths.append((path, avg, sn))

    discarded = set()
    cleaned = defaultdict(list)
    for i, (path, avg, sn) in enumerate(all_paths):
        if tuple(path) in discarded:
            continue
        keep = True
        for j, (other_path, other_avg, _) in enumerate(all_paths):
            if i == j or tuple(other_path) in discarded:
                continue
            for n in path:
                nz  = node_xyz[n, 2]
                nxy = tuple(np.round(node_xyz[n, :2], decimals=4))
                for m in other_path:
                    if abs(node_xyz[m, 2] - nz) < 1e-4:
                        mxy = tuple(np.round(node_xyz[m, :2], decimals=4))
                        if nxy == mxy:
                            if avg < other_avg:
                                keep = False
                                discarded.add(tuple(path))
                            else:
                                discarded.add(tuple(other_path))
                            break
                if not keep:
                    break
        if keep:
            cleaned[sn].append(path)
    return cleaned


# ──────────────────────────────────────────────
# 新实现（与 Apply_momentum.py 中一致）
# ──────────────────────────────────────────────
def cluster_new(gr, node_xyz, threshold=0.5, first_threshold=0.5, layer_centers=None, tolerance=1.0):
    differences = np.abs(gr.x[:, 2].numpy().reshape(-1, 1) - np.array(layer_centers).reshape(1, -1))
    layer_indices = np.argmin(differences, axis=1)
    layers_arr = np.where(np.min(differences, axis=1) <= tolerance, layer_indices, -1)
    layer_tensor = torch.tensor(layers_arr)
    gr.x = torch.cat([gr.x, layer_tensor.unsqueeze(-1)], dim=-1)

    mask = torch.where(
        gr.x[gr.edge_index[0], 2] < 9,
        gr.edge_attr[:, -2] > first_threshold,
        gr.edge_attr[:, -2] > threshold,
    )
    sub_data = torch_geometric.data.Data(
        x=gr.x,
        edge_index=gr.edge_index[:, mask],
        edge_attr=gr.edge_attr[mask, :],
    )

    out_degrees = pyg_utils.degree(sub_data.edge_index[0], sub_data.num_nodes)
    in_degrees  = pyg_utils.degree(sub_data.edge_index[1], sub_data.num_nodes)
    start_nodes = ((out_degrees > 0) & (in_degrees == 0)).nonzero(as_tuple=False).view(-1).numpy()

    import heapq

    edge_index = sub_data.edge_index.cpu().numpy()
    edge_attr  = sub_data.edge_attr.cpu().numpy()
    layers_long = sub_data.x[:, -1].long()
    layers_np   = layers_long.numpy()

    children = defaultdict(list)
    edge_score_dict = defaultdict(dict)
    for i in range(edge_index.shape[1]):
        s, e = int(edge_index[0][i]), int(edge_index[1][i])
        score = float(edge_attr[i][-2])
        children[s].append((e, score))
        edge_score_dict[s][e] = score

    best_score = {}
    parent     = {}

    heap = []
    for sn in start_nodes:
        sn = int(sn)
        best_score[(sn, sn)] = 0.0
        parent[(sn, sn)]     = None
        heapq.heappush(heap, (0.0, sn, sn))

    while heap:
        neg_score, node, start_node = heapq.heappop(heap)
        cum_score = -neg_score
        if cum_score < best_score.get((start_node, node), -float('inf')):
            continue
        node_layer = int(layers_np[node])
        for child, edge_score in children[node]:
            if int(layers_np[child]) != node_layer + 1:
                continue
            new_score = cum_score + edge_score
            key = (start_node, child)
            if new_score > best_score.get(key, -float('inf')):
                best_score[key] = new_score
                parent[key]     = node
                heapq.heappush(heap, (-new_score, child, start_node))

    is_non_leaf = set()
    for (sn, node), par in parent.items():
        if par is not None:
            is_non_leaf.add((sn, par))
    leaves = [(sn, node) for (sn, node) in parent if (sn, node) not in is_non_leaf]

    def reconstruct(sn, leaf):
        path, cur = [], leaf
        while cur is not None:
            path.append(cur)
            cur = parent.get((sn, cur))
        return list(reversed(path))

    final_paths = defaultdict(list)
    for sn, leaf in leaves:
        path = reconstruct(sn, leaf)
        if len(path) < 2:
            continue
        if len(path) > 3:
            total = sum(edge_score_dict[path[i]].get(path[i+1], 0.0) for i in range(len(path)-1))
            avg = total / (len(path) - 1)
        else:
            avg = 0.0
        final_paths[sn].append((path, avg))

    # clean shared hits
    all_paths = []
    for sn, pl in final_paths.items():
        for path, avg in pl:
            if len(path) > 3:
                all_paths.append((path, avg, sn))

    if not all_paths:
        return defaultdict(list)

    hit_to_paths = defaultdict(list)
    for i, (path, _, _) in enumerate(all_paths):
        for n in path:
            xy    = tuple(np.round(node_xyz[n, :2], decimals=4))
            z_key = round(float(node_xyz[n, 2]), 4)
            hit_to_paths[(z_key, xy)].append(i)

    conflicts = defaultdict(set)
    for indices in hit_to_paths.values():
        if len(indices) > 1:
            for a in indices:
                for b in indices:
                    if a != b:
                        conflicts[a].add(b)

    discarded = set()
    cleaned   = defaultdict(list)
    for i, (path, avg, sn) in enumerate(all_paths):
        if i in discarded:
            continue
        keep = True
        for j in conflicts[i]:
            if j in discarded:
                continue
            _, other_avg, _ = all_paths[j]
            if avg < other_avg:
                discarded.add(i)
                keep = False
                break
            else:
                discarded.add(j)
        if keep:
            cleaned[sn].append(path)
    return cleaned


# ──────────────────────────────────────────────
# 合成测试数据生成器
# ──────────────────────────────────────────────
LAYER_CENTERS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

def make_graph(seed, n_tracks=5, shared_hits=True):
    """
    生成一个多径迹合成图：
    - 每条径迹经过全部 8 层（节点 z = layer_center）
    - 部分节点在 shared_hits=True 时会被两条径迹共享（测试 shared-hit 消歧）
    - 每条边有随机置信度分数，在 edge_attr 的倒数第二列（index -2）
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    nodes_x = []   # [x, y, z, dummy]
    edges_src, edges_dst = [], []
    edge_scores = []

    track_node_ids = []  # 每条径迹的节点 id 列表

    for t in range(n_tracks):
        x_offset = rng.uniform(-10, 10)
        y_offset = rng.uniform(-10, 10)
        ids = []
        for layer_idx, z in enumerate(LAYER_CENTERS):
            # 第 2 层故意让 track 0 和 track 1 共享同一个 hit
            if shared_hits and layer_idx == 2 and t == 1:
                ids.append(track_node_ids[0][layer_idx])  # 复用 track0 的节点
            else:
                nid = len(nodes_x)
                nodes_x.append([x_offset + np_rng.uniform(-0.5, 0.5),
                                 y_offset + np_rng.uniform(-0.5, 0.5),
                                 z, 0.0])
                ids.append(nid)
        track_node_ids.append(ids)

    # 建边：同一径迹相邻层之间加边，另外加一些跨径迹干扰边
    for ids in track_node_ids:
        for i in range(len(ids) - 1):
            edges_src.append(ids[i])
            edges_dst.append(ids[i + 1])
            edge_scores.append(np_rng.uniform(0.7, 1.0))  # 高分：真实边

    # 干扰边（低分）
    all_ids = [nid for ids in track_node_ids for nid in ids]
    for _ in range(10):
        a, b = rng.sample(all_ids, 2)
        if a != b:
            edges_src.append(a)
            edges_dst.append(b)
            edge_scores.append(np_rng.uniform(0.01, 0.3))

    n_nodes = len(nodes_x)
    x      = torch.tensor(nodes_x, dtype=torch.float32)
    ei     = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    # edge_attr 格式：[dummy0, dummy1, dummy2, score(-2), dummy(-1)]
    ea     = torch.zeros(len(edge_scores), 5, dtype=torch.float32)
    ea[:, -2] = torch.tensor(edge_scores, dtype=torch.float32)

    # 保证边方向：z_src < z_dst（起点在前）
    valid = x[ei[0], 2] < x[ei[1], 2]
    ei = ei[:, valid]
    ea = ea[valid]

    return torch_geometric.data.Data(x=x, edge_index=ei, edge_attr=ea)


# ──────────────────────────────────────────────
# 比较两个结果是否等价
# ──────────────────────────────────────────────
def normalize(result):
    """将 {start_node: [path, ...]} 转成 frozenset of frozenset，忽略顺序"""
    all_paths = set()
    for paths in result.values():
        for path in paths:
            all_paths.add(tuple(sorted(path)))
    return frozenset(all_paths)


def run_tests():
    n_pass = n_fail = 0
    for seed in range(50):
        for shared in [False, True]:
            gr_orig = make_graph(seed, shared_hits=shared)
            gr_new  = make_graph(seed, shared_hits=shared)  # 相同数据，独立副本

            node_xyz_orig = gr_orig.x[:, :3].cpu().numpy().copy()
            node_xyz_new  = gr_new.x[:, :3].cpu().numpy().copy()

            res_orig = cluster_original(
                gr_orig, node_xyz_orig,
                threshold=0.5, first_threshold=0.5,
                layer_centers=LAYER_CENTERS, tolerance=1.0,
            )
            res_new = cluster_new(
                gr_new, node_xyz_new,
                threshold=0.5, first_threshold=0.5,
                layer_centers=LAYER_CENTERS, tolerance=1.0,
            )

            norm_orig = normalize(res_orig)
            norm_new  = normalize(res_new)

            if norm_orig == norm_new:
                n_pass += 1
            else:
                n_fail += 1
                print(f"[FAIL] seed={seed}, shared={shared}")
                print(f"  original paths : {sorted([sorted(p) for p in norm_orig])}")
                print(f"  new paths      : {sorted([sorted(p) for p in norm_new])}")

    total = n_pass + n_fail
    print(f"\n{'='*50}")
    print(f"Results: {n_pass}/{total} passed, {n_fail}/{total} failed")
    if n_fail == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        exit(1)


def run_benchmark(n_repeats=200):
    """用不同规模的图对两个实现计时"""
    import time

    configs = [
        dict(label="小图  ( 5 tracks, 8 layers, no shared)", n_tracks=5,  shared_hits=False),
        dict(label="小图  ( 5 tracks, 8 layers, shared)",    n_tracks=5,  shared_hits=True),
        dict(label="中图  (15 tracks, 8 layers, no shared)", n_tracks=15, shared_hits=False),
        dict(label="中图  (15 tracks, 8 layers, shared)",    n_tracks=15, shared_hits=True),
    ]

    print(f"\n{'='*65}")
    print(f"{'配置':<38} {'原始(ms/图)':>12} {'新版(ms/图)':>12} {'加速比':>8}")
    print(f"{'-'*65}")

    for cfg_item in configs:
        # 预生成图，避免把数据准备时间算进去
        graphs_orig = [make_graph(seed, n_tracks=cfg_item['n_tracks'], shared_hits=cfg_item['shared_hits'])
                       for seed in range(n_repeats)]
        graphs_new  = [make_graph(seed, n_tracks=cfg_item['n_tracks'], shared_hits=cfg_item['shared_hits'])
                       for seed in range(n_repeats)]
        xyz_orig = [g.x[:, :3].cpu().numpy().copy() for g in graphs_orig]
        xyz_new  = [g.x[:, :3].cpu().numpy().copy() for g in graphs_new]

        # 原始实现计时
        t0 = time.perf_counter()
        for i in range(n_repeats):
            cluster_original(graphs_orig[i], xyz_orig[i],
                             threshold=0.5, first_threshold=0.5,
                             layer_centers=LAYER_CENTERS, tolerance=1.0)
        t_orig = (time.perf_counter() - t0) / n_repeats * 1000  # ms

        # 新实现计时
        t0 = time.perf_counter()
        for i in range(n_repeats):
            cluster_new(graphs_new[i], xyz_new[i],
                        threshold=0.5, first_threshold=0.5,
                        layer_centers=LAYER_CENTERS, tolerance=1.0)
        t_new = (time.perf_counter() - t0) / n_repeats * 1000  # ms

        speedup = t_orig / t_new if t_new > 0 else float('inf')
        print(f"{cfg_item['label']:<38} {t_orig:>12.3f} {t_new:>12.3f} {speedup:>7.2f}x")

    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_tests()
    run_benchmark()
