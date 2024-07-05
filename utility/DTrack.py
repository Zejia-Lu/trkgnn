from typing import Tuple
import numpy as np

import networkx as nx

from utility.Control import cfg

class DTrack:
    def __init__(self):
        self.evt_num = -1
        self.run_num = -1
        self.global_num_trk = -1
        self.no_hits = 0
        self.p_i = -1
        self.p_f = -1
        self.p_avg = -1
        self.p_std = -1
        self.c = 0
        self.c_quality = 0
        self.vertex_hit = None
        self.end_hit = None
        self.p_dir = None

        self.track_id = -1
        self.has_vertex = []
        self.vertex_z = []
        self.track_2_id = []

        self.full_track = 0

        self.eps = 2.0  # mm

    def __repr__(self):
        return \
            f"DTrack [{self.no_hits} hits]: {self.p_avg:.4f} E/E0"

    def __str__(self):
        return \
            f"DTrack: evt_num={self.evt_num}, run_num={self.run_num}, no_hits={self.no_hits}, " \
            f"p_i={self.p_i}, p_f={self.p_f}, p_avg={self.p_avg}, " \
            f"vertex_hit={self.vertex_hit}, end_hit={self.end_hit}"

    def from_graph(self, graph: nx.Graph, e0=1.0, tracker_boundary: Tuple[float, float] = None):
        self.no_hits = len(graph.nodes)
        self.evt_num = graph.graph['evt_num']
        self.run_num = graph.graph['run_num']

        # sort nodes by z in ascending order
        nodes_order = sorted(graph.nodes(), key=lambda n: graph.nodes[n]['z'])
        if cfg['momentum_predict']:
            # Find the edge with the maximum 'p_pred' attribute for the first node
            self.p_i = max(graph.edges(nodes_order[0], data=True), key=lambda x: x[2]['p_pred'])[2]['p_pred'] * e0
            # Find the edge with the minimum 'p_pred' attribute for the last node
            self.p_f = min(graph.edges(nodes_order[-1], data=True), key=lambda x: x[2]['p_pred'])[2]['p_pred'] * e0
        else:
            self.p_i = 0
            self.p_f = 0

        self.vertex_hit = graph.nodes[nodes_order[0]]
        self.end_hit = graph.nodes[nodes_order[-1]]

        # full track: the vertex node and the end node are at the boarder of the detector
        if tracker_boundary is not None:
            vertex_good = abs(self.vertex_hit['z'] - tracker_boundary[0]) < self.eps
            end_good = abs(self.end_hit['z'] - tracker_boundary[1]) < self.eps

            if vertex_good and end_good:
                self.full_track = 1
            elif vertex_good and not end_good:
                self.full_track = 2
            elif end_good and not vertex_good:
                self.full_track = 3
            else:
                self.full_track = 0
