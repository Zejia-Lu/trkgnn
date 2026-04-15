import gc
import logging
import pickle
import os
import itertools
import torch
import torch_geometric
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from collections import deque, defaultdict
from copy import deepcopy

from utility.Control import cfg
from utility.DataLoader import get_data_loaders
from utility.EverythingNeeded import config_logging, build_model
from utility.FunctionTime import timing_decorator
from utility.DTrack import DTrack


# from pyvis.network import Network


def setup():
    # check OS
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available.")

        # Check the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}, but only 1 is used.")

        device = torch.device("cuda:0")
    # elif torch.has_mps:
    #     print("mps is available.")
    #     device = torch.device("mps")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    cfg['device'] = device
    # cfg['data']['read_from_graph'] = False
    cfg['momentum_predict'] = True

    # Run all events
    if 'global_stop' in cfg['data']:
        del cfg['data']['global_stop']


@timing_decorator
@torch.no_grad()
def predict(input_dir: list[str], model_dir: str, output_dir: str, truth: bool = False, vtx_model=None):
    config_logging(False, output_dir=output_dir, prefix='apply')
    setup()
    logger = logging.getLogger("Apply.Momentum")

    logger.info(f"input_dir: {input_dir}")
    logger.info(f"model_dir: {model_dir}")
    logger.info(f"output_dir: {output_dir}")
    if truth: logger.info(f"Truth Mode !")

    # Build model and load model
    model = build_model(cfg['device'], distributed=False)
    tar = torch.load(model_dir, map_location=cfg['device'])
    model.load_state_dict(tar['model'])
    model.to(cfg['device'])
    model.eval()

    # Build data loader
    for i, data_dir in enumerate(input_dir):
        logger.info(f"Processing {i + 1}/{len(input_dir)}: {data_dir}")

        data_generator = get_data_loaders(
            data_dir,
            chunk_size=cfg['data']['chunk_size'],
            batch_size=cfg['data']['batch_size'],
            distributed=False,
            n_workers=cfg['data']['n_workers'],
            shuffle=True,
            apply=True,
        )

        output_graph_dir = os.path.join(output_dir, cfg['data']['collection'])
        os.makedirs(output_graph_dir, exist_ok=True)

        itr = 0
        while True:
            try:
                next_chunk = next(data_generator)
                if isinstance(next_chunk, tuple):
                    apply_loader, chunk_name = next_chunk
                else:
                    apply_loader, chunk_name = next_chunk, None
                if chunk_name:
                    logger.info(f"Chunk source file: {chunk_name}")
                logger.info(f"Processing {itr + 1}th iteration with {len(apply_loader)} batches.")

                predicted_graph_list = []
                analyzed_tracks_list = []
                num_graphs = 0
                # Loop over batches
                for j, batch in enumerate(apply_loader):
                    batch = batch.to(cfg['device'])
                    p_pred_all, y_pred = model(batch)
                    # y_pred = torch.sigmoid(new_y_score)
                    batch.edge_attr = torch.cat([batch.edge_attr, y_pred.unsqueeze(-1), p_pred_all.unsqueeze(-1)],
                                                dim=1)
                    batch = batch.to("cpu")

                    if 'tracking_layers' in cfg['data']:
                        num_batches = 0
                        for idx in range(batch.num_graphs):
                            paths = cluster(batch[idx], threshold=cfg['data']['threshold'], first_threshold=cfg['data']['first_threshold'])
                            analyzed_tracks_list.append(analyze_tracks(batch[idx], paths, vtx_model=vtx_model))
                            num_batches += 1
                        logger.debug(
                            f"Number of graphs: {num_batches} processed in {j}th batch. Length of batch: {len(batch)}")

                    predicted_graph_list.extend(batch.to_data_list())
                    num_graphs += batch.num_graphs

                if chunk_name:
                    stem, ext = os.path.splitext(chunk_name)
                    momentum_name = chunk_name
                    tracks_name = f"{stem}_tracks.lt"
                else:
                    momentum_name = f"momentum_{itr}.pt"
                    tracks_name = f"tracks_{itr}.lt"

                torch.save(predicted_graph_list, os.path.join(output_graph_dir, momentum_name))
                torch.save(analyzed_tracks_list, os.path.join(output_graph_dir, tracks_name))

                logger.info(f"Number of graphs: {num_graphs} processed in {itr + 1}th iteration.")
                itr += 1
            except StopIteration:
                print("Finish")
                # logout to root file
                logger.info(f"Finish predicting file: {data_dir}")
                break

# BFS Algorithm
# @timing_decorator
# def cluster(gr: torch_geometric.data.Data, threshold: float = 0.5, first_threshold: float = 0.5):
#     logger = logging.getLogger("Apply.Momentum.Cluster")

#     @timing_decorator
#     def classify_layers(z_values, layer_centers, tolerance):
#         # Calculate the absolute difference between z-values and layer centers
#         differences = np.abs(z_values - np.array(layer_centers).reshape(1, -1))
#         # Find the index of the minimum difference
#         layer_indices = np.argmin(differences, axis=1)
#         # Determine if the minimum difference is within the tolerance, otherwise set to -1 or some null value
#         layers = np.where(np.min(differences, axis=1) <= tolerance, layer_indices, -1)

#         return layers

#     layer = classify_layers(
#         z_values=gr.x[:, 2].numpy().reshape(-1, 1),
#         layer_centers=cfg['data']['tracking_layers'],
#         tolerance=1.0
#     )

#     gr.x = torch.cat([gr.x, torch.tensor(layer).unsqueeze(-1)], dim=-1)

#     mask = torch.where(
#         gr.x[gr.edge_index[0], 2] < 9,  # For the first edge
#         gr.edge_attr[:, -2] > first_threshold,  
#         gr.edge_attr[:, -2] > threshold  # For the other edges
#     )

#     # Create the subgraph
#     sub_data = torch_geometric.data.Data(
#         x=gr.x,
#         edge_index=gr.edge_index[:, mask],
#         edge_attr=gr.edge_attr[mask, :],
#     )

#     out_degrees = pyg_utils.degree(sub_data.edge_index[0], sub_data.num_nodes)
#     in_degrees = pyg_utils.degree(sub_data.edge_index[1], sub_data.num_nodes)

#     # find initial start nodes with only out-edges and no in-edges
#     start_nodes = (out_degrees > 0) & (in_degrees == 0)
#     start_nodes = start_nodes.nonzero(as_tuple=False).view(-1).numpy()

#     @timing_decorator
#     def bfs_for_multiple_starts_with_layers(data, start_nodes_, layers):
#         paths_from_start = defaultdict(list)
#         queue = deque([(start_node, [start_node]) for start_node in start_nodes_])  # (current_node, path)

#         edge_index = data.edge_index.cpu().numpy()

#         # Create a children dictionary to avoid recomputing it
#         children = defaultdict(list)
#         for start, end in edge_index.T:
#             children[start].append(end)

#         # Track used nodes to avoid reprocessing
#         used_nodes = set()  # Set of tuples (x, y, z)

#         while queue:
#             current_node, path = queue.popleft()
            
#             # Skip if the current node is already used
#             current_coords = tuple(gr.x[current_node, :3].cpu().numpy())  # Convert to tuple for hashing

#             # Skip if the current coordinates are already used
#             if current_coords in used_nodes:
#                 continue

#             current_layer = layers[current_node].item()
#             for child in children[current_node]:
#                 child_layer = layers[child].item()
#                 # Only add the child if it's in the next layer
#                 if child_layer == current_layer + 1:

#                     index_to_modify = next(
#                         (i for i, sublist in enumerate(paths_from_start[path[0]]) if sublist == path),
#                         None
#                     )

#                     if index_to_modify is not None:
#                         paths_from_start[path[0]][index_to_modify].append(child)
#                     else:
#                         index_to_modify = len(paths_from_start[path[0]])
#                         paths_from_start[path[0]].append([*path, child])
                    
#                     # Check if the path has more than 3 nodes
#                     # If so, mark all nodes in the path as "used"
#                     if len(paths_from_start[path[0]][index_to_modify]) > 3:
#                         for node in paths_from_start[path[0]][index_to_modify]:
#                             node_coords = tuple(gr.x[node, :3].cpu().numpy())
#                             used_nodes.add(node_coords)

#                     queue.append((child, deepcopy(paths_from_start[path[0]][index_to_modify])))

#         return paths_from_start

#     layers = sub_data.x[:, -1].long()
#     paths_from_start = bfs_for_multiple_starts_with_layers(sub_data, start_nodes, layers)
        
#     # 5. 筛选 paths_from_start 处理 shared hit
#     def clean_shared_hits(paths_from_start):
#         all_paths = []
#         for start_node, path_list in paths_from_start.items():
#             for path in path_list:
#                 if len(path) > 3:  # 只处理长度大于 3 的路径
#                     all_paths.append((path, start_node))

#         cleaned_paths_from_start = defaultdict(list)
#         discarded_paths = set()  # 记录被丢弃的路径

#         for i, (path, start_node) in enumerate(all_paths):
#             if tuple(path) in discarded_paths:
#                 continue  # 路径已被丢弃，跳过

#             keep_path = True
#             for j, (other_path, other_start_node) in enumerate(all_paths):
#                 if i == j or tuple(other_path) in discarded_paths:
#                     continue  # 跳过自身或其他已被丢弃的路径

#                 # 检查路径上的每个节点
#                 for n in path:
#                     node_coords = gr.x[n, :3].cpu().numpy()
#                     node_z = node_coords[2]
#                     node_xy = tuple(np.round(node_coords[:2], decimals=4))  # 只比较 x, y

#                     # 在 other_path 中查找相同 z 坐标的节点
#                     for m in other_path:
#                         other_coords = gr.x[m, :3].cpu().numpy()
#                         other_z = other_coords[2]
#                         if abs(other_z - node_z) < 1e-4:  # 相同 z 坐标
#                             other_xy = tuple(np.round(other_coords[:2], decimals=4))
#                             if node_xy == other_xy:  # Shared hit
#                                 discarded_paths.add(tuple(other_path)) 
#                                 break  # 找到 shared hit 后无需检查其他节点
#                     if not keep_path:
#                         break  # 当前路径已被丢弃，退出外层循环

#             if keep_path:
#                 cleaned_paths_from_start[start_node].append(path)

#         for start_node, path_list in cleaned_paths_from_start.items():
#             for path in path_list:
#                 logger.debug(f"Cleaned path from start_node {start_node}: {len(path)} nodes")

#         return cleaned_paths_from_start 

#     cleaned_paths_from_start = clean_shared_hits(paths_from_start)

#     return cleaned_paths_from_start

# Dijkstra Algorithm
@timing_decorator
def cluster(gr: torch_geometric.data.Data, threshold: float = 0, first_threshold: float = 0):
    logger = logging.getLogger("Apply.Momentum.Cluster")

    @timing_decorator
    def classify_layers(z_values, layer_centers, tolerance):
        # Calculate the absolute difference between z-values and layer centers
        differences = np.abs(z_values - np.array(layer_centers).reshape(1, -1))
        # Find the index of the minimum difference
        layer_indices = np.argmin(differences, axis=1)
        # Determine if the minimum difference is within the tolerance, otherwise set to -1 or some null value
        layers = np.where(np.min(differences, axis=1) <= tolerance, layer_indices, -1)

        return layers

    layer = classify_layers(
        z_values=gr.x[:, 2].numpy().reshape(-1, 1),
        layer_centers=cfg['data']['tracking_layers'],
        tolerance=1.0
    )

    gr.x = torch.cat([gr.x, torch.tensor(layer).unsqueeze(-1)], dim=-1)

    mask = torch.where(
        gr.x[gr.edge_index[0], 2] < 9,  # For the first edge
        gr.edge_attr[:, -2] > first_threshold,  
        gr.edge_attr[:, -2] > threshold  # For the other edges
    )
    # print(f"mask: {mask}")
    # Create the subgraph
    sub_data = torch_geometric.data.Data(
        x=gr.x,
        edge_index=gr.edge_index[:, mask],
        edge_attr=gr.edge_attr[mask, :],
    )

    out_degrees = pyg_utils.degree(sub_data.edge_index[0], sub_data.num_nodes)
    in_degrees = pyg_utils.degree(sub_data.edge_index[1], sub_data.num_nodes)

    # find initial start nodes with only out-edges and no in-edges
    start_nodes = (out_degrees > 0) & (in_degrees == 0)
    start_nodes = start_nodes.nonzero(as_tuple=False).view(-1).numpy()

    # Precompute node coordinates once to avoid repeated tensor→numpy conversions
    node_xyz = gr.x[:, :3].cpu().numpy()

    @timing_decorator
    def dijkstra_with_parent_pointers(data, start_nodes_, layers):
        import heapq

        edge_index = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()
        layers_arr = layers.numpy()

        # Build adjacency list and edge-score lookup in one pass
        children = defaultdict(list)
        edge_score_dict = defaultdict(dict)
        for i in range(edge_index.shape[1]):
            s, e = int(edge_index[0][i]), int(edge_index[1][i])
            score = float(edge_attr[i][-2])
            children[s].append((e, score))
            edge_score_dict[s][e] = score

        # Dijkstra with parent pointers, keyed by (start_node, node).
        # This correctly handles shared hits: the same node C can appear in
        # multiple start nodes' trees independently, each with its own best
        # score and parent. Heap entries are (neg_score, node, start_node) —
        # no path list copying, so each push is O(1) instead of O(depth).
        best_score = {}   # (start_node, node) -> best cumulative score
        parent = {}       # (start_node, node) -> predecessor node (None for roots)

        heap = []
        for sn in start_nodes_:
            sn = int(sn)
            best_score[(sn, sn)] = 0.0
            parent[(sn, sn)] = None
            heapq.heappush(heap, (0.0, sn, sn))  # (neg_score, node, start_node)

        while heap:
            neg_score, node, start_node = heapq.heappop(heap)
            cum_score = -neg_score

            if cum_score < best_score.get((start_node, node), -float('inf')):
                continue

            node_layer = int(layers_arr[node])
            for child, edge_score in children[node]:
                if int(layers_arr[child]) != node_layer + 1:
                    continue
                new_score = cum_score + edge_score
                key = (start_node, child)
                if new_score > best_score.get(key, -float('inf')):
                    best_score[key] = new_score
                    parent[key] = node
                    heapq.heappush(heap, (-new_score, child, start_node))

        # Leaf nodes per start_node: (sn, node) pairs where no child in sn's
        # tree points back to node as its parent.
        is_non_leaf = set()
        for (sn, node), par in parent.items():
            if par is not None:
                is_non_leaf.add((sn, par))
        leaves = [(sn, node) for (sn, node) in parent if (sn, node) not in is_non_leaf]

        def reconstruct_path(sn, leaf):
            path = []
            cur = leaf
            while cur is not None:
                path.append(cur)
                cur = parent.get((sn, cur))
            return list(reversed(path))

        final_paths = defaultdict(list)
        for sn, leaf in leaves:
            path = reconstruct_path(sn, leaf)
            if len(path) < 2:
                continue
            if len(path) > 3:
                total = sum(edge_score_dict[path[i]].get(path[i + 1], 0.0)
                            for i in range(len(path) - 1))
                avg_score = total / (len(path) - 1)
            else:
                avg_score = 0.0
            final_paths[sn].append((path, avg_score))

        return final_paths

    layers = sub_data.x[:, -1].long()
    paths_from_start = dijkstra_with_parent_pointers(sub_data, start_nodes, layers)

    # Filter paths_from_start to resolve shared hits
    def clean_shared_hits(paths_from_start):
        all_paths = []
        for start_node, path_list in paths_from_start.items():
            for path, avg_score in path_list:
                if len(path) > 3:
                    all_paths.append((path, avg_score, start_node))

        if not all_paths:
            return defaultdict(list)

        # Build spatial index: (z_rounded, xy_rounded) -> [path indices]
        # Replaces the O(N² × path_len) nested loop with an O(total_hits) pass.
        hit_to_paths = defaultdict(list)
        for i, (path, _, _) in enumerate(all_paths):
            for n in path:
                xy = tuple(np.round(node_xyz[n, :2], decimals=4))
                z_key = round(float(node_xyz[n, 2]), 4)
                hit_to_paths[(z_key, xy)].append(i)

        # Collect conflict pairs from shared hits
        conflicts = defaultdict(set)
        for indices in hit_to_paths.values():
            if len(indices) > 1:
                for a in indices:
                    for b in indices:
                        if a != b:
                            conflicts[a].add(b)

        discarded = set()
        cleaned_paths_from_start = defaultdict(list)

        for i, (path, avg_score, start_node) in enumerate(all_paths):
            if i in discarded:
                continue
            keep = True
            for j in conflicts[i]:
                if j in discarded:
                    continue
                _, other_avg, _ = all_paths[j]
                if avg_score < other_avg:
                    discarded.add(i)
                    keep = False
                    break
                else:
                    discarded.add(j)
            if keep:
                cleaned_paths_from_start[start_node].append(path)

        for start_node, path_list in cleaned_paths_from_start.items():
            for path in path_list:
                logger.debug(f"Cleaned path from start_node {start_node}: {len(path)} nodes")

        return cleaned_paths_from_start

    cleaned_paths_from_start = clean_shared_hits(paths_from_start)
    return cleaned_paths_from_start


@timing_decorator
def analyze_tracks(graph: torch_geometric.data.Data, paths: dict[list], vtx_model=None):
    logger = logging.getLogger("Apply.Momentum.Analyze")
    trajectories = []
    traj_graphs = defaultdict(list)
    edge_set = set(map(tuple, graph.edge_index.T.tolist()))
    for idx, start_points in enumerate(paths.keys()):
        for path in paths[start_points]:
            traj = DTrack()
            if isinstance(path, torch.Tensor):
                path = path.tolist()
            elif isinstance(path, np.ndarray):
                path = path.astype(int).tolist()
            # 确保每个元素是 int
            path = [int(n) for n in path]
            if not isinstance(path, list) or not all(isinstance(n, int) for n in path):
                logger.warning(f"Illegal path format: {path}")
                continue

            # 检查路径长度
            if len(path) < 2:
                # logger.warning(f"Path {path} has less than 2 nodes, skipping...")
                continue

            # 检查节点索引有效性
            max_node = max(path)
            if max_node >= graph.num_nodes or max_node < 0:
                # logger.warning(f"Invalid node index {max_node} in path {path}")
                continue

            try:
                sub_edges_index, sub_edges = subgraph(
                    subset=path,
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    relabel_nodes=True
                )
            except Exception as e:
                logger.error(f"Failed to process path {path}: {str(e)}")
                continue

            # 检查 sub_edges 是否为空
            if sub_edges.shape[0] == 0:
                # logger.warning(f"Path {path} has no edges, skipping...")
                continue

            # sub_edges_index, sub_edges = subgraph(path, graph.edge_index, graph.edge_attr, relabel_nodes=True)

            # traj_graphs.append(torch_geometric.data.Data(
            #     x=torch.from_numpy(np.array(range(len(path)))),
            #     edge_index=sub_edges_index,
            #     edge_attr=sub_edges[:,[0,1,2,-1]],
            # ))

            if vtx_model is not None:
                traj_graphs['x'].append(graph.x[path])
                traj_graphs['edge_index'].append(sub_edges_index)
                traj_graphs['edge_attr'].append(sub_edges[:, [0, 1, 2, -1]])

            traj.run_num = graph.run_num.item()
            traj.evt_num = graph.evt_num.item()
            traj.global_num_trk = graph.n.item()
            traj.track_id = idx

            traj.no_hits = len(path)
            traj.p_i = sub_edges[0, -1].item()
            traj.p_f = sub_edges[-1, -1].item()
            
            # Record all 'p_pred' values for edges connected to each node
            # traj.p_all = sub_edges[:, -1].numpy().tolist()

            p_vals = sub_edges[:, -1].numpy()
            traj.p_avg = np.mean(p_vals)
            traj.p_std = np.std(p_vals)

            all_hit_coords = graph.x[path, :3].numpy()
            traj.first_hit = all_hit_coords[0]
            traj.end_hit = all_hit_coords[-1]
            traj.all_hits = all_hit_coords.tolist()

            # Check if there are at least 4 connected hits (3 consecutive edges)
            if len(path) >= 4:
                connected_count = 0
                for i in range(len(path) - 1):
                    if (path[i], path[i + 1]) in edge_set:
                        connected_count += 1
                        if connected_count == 3:  # 3 consecutive edges mean 4 connected hits
                            traj.above_four_hits = 1
                            break
                    else:
                        connected_count = 0  # Reset if not consecutive
            else:
                traj.above_four_hits = 0
                
            # calculate charge
            if len(path) < 3:
                traj.c = 0
                traj.c_quality = 0
            else:
                dx = np.diff(graph.x[path, 0].numpy())
                ddx = np.diff(dx)
                if np.all(ddx > 0):
                    traj.c = 1
                    traj.c_quality = 1
                elif np.all(ddx < 0):
                    traj.c = -1
                    traj.c_quality = 1
                else:
                    signs = np.sign(ddx)
                    # Make sure the signs array is of integer type
                    signs = signs.astype(np.int64)
                    counts = np.bincount(signs + 1)
                    if len(np.unique(counts)) == len(counts):
                        traj.c = np.argmax(counts) - 1
                    else:
                        traj.c = 0
                    traj.c_quality = 0

            trajectories.append(traj)

    if vtx_model is not None:
        for x1, x2 in list(itertools.combinations(range(len(trajectories)), 2)):
            s_idx = len(traj_graphs['x'][x1])
            gr = torch_geometric.data.Data(
                x=torch.cat((traj_graphs['x'][x1], traj_graphs['x'][x2] + s_idx), dim=0),
                edge_index=torch.cat((traj_graphs['edge_index'][x1], traj_graphs['edge_index'][x2] + s_idx), dim=1),
                edge_attr=torch.cat((traj_graphs['edge_attr'][x1], traj_graphs['edge_attr'][x2]), dim=0),
            )

            gr.to(cfg['device'])

            cls, reg = vtx_model(gr)

            v_pred = torch.sigmoid(cls)
            z_pred = reg

            trajectories[x1].has_vertex.append(v_pred.item())
            trajectories[x2].has_vertex.append(v_pred.item())

            trajectories[x1].vertex_z.append(z_pred.item())
            trajectories[x2].vertex_z.append(z_pred.item())

            trajectories[x1].track_2_id.append(x2)
            trajectories[x2].track_2_id.append(x1)

    return trajectories
