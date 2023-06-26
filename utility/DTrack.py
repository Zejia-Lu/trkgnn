import networkx as nx


class DTrack:
    def __init__(self):
        self.evt_num = -1
        self.run_num = -1
        self.no_hits = 0
        self.p_i = -1
        self.p_f = -1
        self.vertex_hit = None
        self.end_hit = None

        pass

    def from_graph(self, graph: nx.Graph, e0=1.0):
        self.no_hits = len(graph.nodes)
        self.evt_num = graph.graph['evt_num']
        self.run_num = graph.graph['run_num']

        # sort nodes by z in ascending order
        nodes_order = sorted(graph.nodes(), key=lambda n: graph.nodes[n]['z'])
        # Find the edge with the maximum 'p_pred' attribute for the first node
        self.p_i = max(graph.edges(nodes_order[0], data=True), key=lambda x: x[2]['p_pred'])[2]['p_pred'] * e0
        # Find the edge with the minimum 'p_pred' attribute for the last node
        self.p_f = min(graph.edges(nodes_order[-1], data=True), key=lambda x: x[2]['p_pred'])[2]['p_pred'] * e0

        self.vertex_hit = graph.nodes[nodes_order[0]]
        self.end_hit = graph.nodes[nodes_order[-1]]
