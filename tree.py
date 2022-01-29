from time import time_ns
from colour import Color
from manim import *
from typing import Tuple, Union
import networkx as nx
from numpy import random

seed = time_ns()
seed =1643069429145020999
print(seed)
rnd = random.default_rng(seed)

class Node():
    def __init__(self, id: str, is_in_monte_carlo: bool, is_dead_end: bool, turn: bool, action: str, parent=None, probs=[]):
        self.is_in_monte_carlo = is_in_monte_carlo
        self.is_dead_end = is_dead_end
        self.turn = turn
        self.id = id
        self.action = action
        self.parent = parent
        self.probs = probs
        self.value = rnd.random() * 100

    def __repr__(self):
        return f"{self.id}: {self.is_in_monte_carlo}, {self.is_dead_end}"


class Tree():
    def __init__(self) -> None:
        self.g = nx.DiGraph()

    def add_child(self, node: Node, parent: Union[Node, None], prob):
        node.parent = parent
        self.g.add_node(node.id, data=node)
        if parent:
            self.g.add_edge(parent.id, node.id, prob=prob)

    def generate_tree(self, n: int, parent: Union[Node, None]):
        if not n:
            return self
        turn = False if not parent else not parent.turn
        if parent == None:
            node = Node("root", False, False, False, "raise", parent)
            self.add_child(node, parent, 1)
            self.generate_tree(n-1, node)
            return self
        if parent.action == "raise":
            probs = rnd.random(3)
            probs /= probs.sum()
            parent.probs = probs
            for i, action in enumerate(["call", "raise", "fold"]):
                node = Node(f"{n}-{action}-{parent.id}", False,
                            False, turn, action, parent)
                self.add_child(node, parent, probs[i])
                self.generate_tree(n-1, node)

        if parent.action == "call":
            probs = rnd.random(2)
            probs /= probs.sum()
            parent.probs = probs
            for i, action in enumerate(["check", "raise"]):
                node = Node(f"{n}-{action}-{parent.id}", False,
                            False, turn, action, parent)
                self.add_child(node, parent, probs[i])
                self.generate_tree(n-1, node)

        if parent.action == "check":
            probs = rnd.random(2)
            probs /= probs.sum()
            parent.probs = probs
            for i, action in enumerate(["check", "raise"]):
                node = Node(f"{ n }-{action}-{parent.id}", False,
                            False, turn, action, parent)
                self.add_child(node, parent, probs[i])
                self.generate_tree(n-1, node)
        if parent.action == "fold":
            parent.probs = [1]
            parent.is_dead_end = True
            return self
        return self

    def set_value_from_children(self, node, callback = None):
        node_t = self.node_data(node)

        if len(self.g.out_edges(node)):
            avg = 0
            for i, e in enumerate(self.g.out_edges(node)):
                v = e[1]
                self.set_value_from_children(v, callback)
                avg += self.node_data(v).value * node_t.probs[i]

            if node_t.value != avg:
                node_t.value = avg
                if callback:
                    callback(node, avg)

        return node_t.value

    def monte_carlo(self, n, callback=None, traversed=[]):
        node: Node = self.node_data(n)
        node.is_in_monte_carlo = True
        edges = self.g.out_edges(n)
        traversed.append(node)
        if len(edges):
            edge = rnd.choice(
                list(map(lambda x: x[1], edges)), p=node.probs)
            # edge = list(edges)[0][1]
            if callback:
                callback(node, (node.id, edge), len(traversed))
            self.monte_carlo(edge, callback, traversed)
        elif callback:
            callback(node, None, len(traversed))
        return traversed

    def node_data(self, node) -> Node:
        return self.g.nodes(data=True)[node]['data']


class LabeledLine(VGroup):
    def __init__(self, start, end, ll: str, **args):
        self.l = Line(start, end, **args)
        super().__init__(Tex(ll).move_to(self.l).shift(
            DOWN, 0.001), Line(start, end, **args))

    def get_start(self, **args):
        return self.l.get_start(**args)

    def get_end(self, **args):
        return self.l.get_end(**args)


class TreeMobject(Graph):
    def __init__(self,  tree: Tree, node_size=0.3) -> None:

        self.tree = tree
        self.node_size = node_size
        nodes = list(map(lambda x: x[1]["data"], tree.g.nodes.items()))
        edges = list(tree.g.edges)

        args = {
            "vertices": list(map(lambda x: x.id, nodes)),
            "vertex_config": self.node_to_style(nodes),
            "vertex_type": Circle,
            "edges": edges,
            "edge_type": Line,
            "layout": "tree",
            "root_vertex": 'root',
            "layout_scale": (3, 1.8),
            "edge_config": self.edge_to_style(edges),
        }
        super().__init__(**args)


        self.node_to_value: Dict[str, Text] = {}
        values = []
        for v in self.tree.g.nodes:
            value_label = Text(font_size=DEFAULT_FONT_SIZE / 3,
                                        text=str(int(self.tree.node_data(v).value)), color=WHITE).move_to(self[v])
            self.node_to_value[v] = value_label
            values += [value_label]

        self.values = values


    def add_vertices(self, *nodes):
        return super().add_vertices(*list(map(lambda x: x.id, nodes)), vertex_config=self.node_to_style(nodes), vertex_type=Circle)

    def node_to_style(self, nodes: List[Node]):

        return {node.id: {
                "radius": self.node_size,
                "color": BLUE if node.turn and node.is_in_monte_carlo else DARK_BLUE if node.turn and not node.is_in_monte_carlo else GREEN if node.is_in_monte_carlo else GREEN_D,
                "fill_opacity": 1,
                "stroke_color": GRAY if node.is_in_monte_carlo else None
                } for node in nodes}

    def edge_to_style(self, edges):
        if not self.tree:
            return None
        return {
            k: {
                "color": GRAY if not self.tree.node_data(k[1]).is_in_monte_carlo else RED,
            }
            for k in edges

        }

    def animate_node(self, node):
        return self[node].animate

    def reevaluate_callback(self, scene):
        def ree(node, value):
            scene.play(Wiggle(self.node_to_value[node]))
            new_text = Text(font_size=DEFAULT_FONT_SIZE / 3,
                                            text=str(int(value)), color=WHITE).move_to(self[node])
            scene.play(Transform(self.node_to_value[node], new_text))
        return ree


    def monte_carlo_callback(self, scene):
        def re(node, edge, n):
            scene.play(self.animate_node(node.id).set_color(RED), run_time=0.6)
            if edge:
                scene.play(self.edges[edge].animate.set_color(RED), run_time=0.6)
        return re

    def show_each_player(self, scene):
        blue_nodes = [self[n[1]['data'].id]
            for n in self.tree.g.nodes(data=True) if n[1]['data'].turn]
        green_nodes = [self[n[1]['data'].id]
            for n in self.tree.g.nodes(data=True) if not n[1]['data'].turn]

        scene.wait(1)
        scene.play(*map(lambda x: Wiggle(x, scale_value=1.3), blue_nodes), run_time=1.91)
        scene.wait(duration=0.2)
        scene.play(*map(lambda x: Wiggle(x, scale_value=1.3), green_nodes), run_time=1.91)

    def add_graph(self, scene):
        scene.play(Create(self), duration=3)
        scene.wait(duration=1)

    def monte_carlo(self, scene):
         return self.tree.monte_carlo('root', callback=self.monte_carlo_callback(scene))


    def zoom_to_decision(self, decision, out_edges, scene):
        return scene.camera.auto_zoom(
            [self[node] for _, node in out_edges] + [self[decision.id]], margin=1.5)

    def update_labels(self):
        for n, l in self.node_to_value.items():
            l.move_to(self[n])
    def add_labels(self,):
        return [Create(l) for l in self.values]


class GrayCamera(MovingCamera):
    def __init__(self, background_color="#1E1E1E", **kwargs):
        super().__init__(background_color=background_color, **kwargs)




class TestTree(MovingCameraScene):
    def __init__(self, camera_class=..., **kwargs):
        # initializing tree
        self.tree = Tree()
        self.tree.generate_tree(3, None)
        self.tree.set_value_from_children('root')

        # initializing graph

        super().__init__(camera_class=GrayCamera)
        self.graph = TreeMobject(self.tree, node_size=0.3)

    def construct(self):
        # self.true_construct()
        # title = Title(r"Minimização de arrependimento \\contrafactual de Monte Carlo")
        # self.play(Write(title, run_time=4))
        # self.wait(duration=2)
        # self.play(FadeOut(title))
        self.wait(duration=9.3)
        self.graph.add_graph(self)
        self.wait(0.8)


        self.play(*[Wiggle(node) for node in self.graph.vertices.values()], run_time=1.5)
        self.wait(1)
        self.play(*[edge.animate.set_color(WHITE) for edge in self.graph.edges.values()], run_time=0.4)
        self.play(*[edge.animate.set_color(GRAY) for edge in self.graph.edges.values()], run_time=0.4)

        self.wait(8)

        self.graph.show_each_player(self)

        self.wait(5)

        nodes = self.graph.monte_carlo(self)
        self.true_construct(nodes)

    def true_construct(self, nodes):

        self.wait(2)
        self.play(*[self.graph.edges[k].animate.set_stroke_width(np.round(self.tree.g.edges[k]['prob'], 2)*15) for k in self.tree.g.edges])

        self.wait(5)

        n=2
        decision = nodes[-n]
        out_edges : List[Tuple[str, str]] = list(self.tree.g.out_edges(decision.id))

        choice = ""
        for edge in out_edges:
            for node in nodes:
                if node.id == edge[1]:
                    choice = node.id



        self.play(self.graph.zoom_to_decision(decision, out_edges, self), *self.graph.add_labels())
        self.play(Wiggle(self.graph[choice]))
        self.wait(1.5)

        vertices = []
        edges = []
        vs = []

        for u, v in out_edges:
            node_data = self.tree.node_data(v)
            vertices += [v]
            edges += [(u, v)]
            vs += [node_data.value]


        for _, v in out_edges:
            if v != choice:
                self.play(Wiggle(self.graph[v]), run_time=1)


        vs=np.array(vs)
        s = vs.sum()
        vs /= vs.sum()
        probs = decision.probs
        for i in range(len(probs)):
            probs[i] += vs[i] - decision.value / s
            probs[i] = max(0, probs[i])
        probs /= probs.sum()


        edge_ani=[]
        for i, edge in enumerate(edges):
            edge_ani += [self.graph.edges[edge]
                .animate.set_stroke_width(probs[i] * 15)]
        self.play(*edge_ani, run_time=2)

        decision.probs = probs
        self.wait(9)
        self.tree.set_value_from_children(decision.id, callback=self.graph.reevaluate_callback(self))


        self.wait(3)
        for n in range(3, len(nodes) + 1):
            decision = nodes[-n]
            out_edges : List[Tuple[str, str]] = list(self.tree.g.out_edges(decision.id))

            choice = ""
            for edge in out_edges:
                for node in nodes:
                    if node.id == edge[1]:
                        choice = node.id



            self.play(self.graph.zoom_to_decision(decision, out_edges, self))

            vertices = []
            edges = []
            vs = []

            for u, v in out_edges:
                node_data = self.tree.node_data(v)
                vertices += [v]
                edges += [(u, v)]
                vs += [node_data.value]


            for _, v in out_edges:
                self.play(Wiggle(self.graph[v]), run_time=0.3)


            vs=np.array(vs)
            s = vs.sum()
            vs /= vs.sum()
            probs = decision.probs
            for i in range(len(probs)):
                probs[i] += vs[i] - decision.value / s
                probs[i] = max(0, probs[i])
            probs /= probs.sum()


            edge_ani=[]
            for i, edge in enumerate(edges):
                edge_ani += [self.graph.edges[edge]
                    .animate.set_stroke_width(probs[i] * 15)]
            self.play(*edge_ani, run_time=1)

            decision.probs = probs
            self.tree.set_value_from_children(decision.id, callback=self.graph.reevaluate_callback(self))
        self.play(self.camera.auto_zoom(self.graph, margin=2))
        self.play(FadeOut(self.graph), *[FadeOut(l) for l in self.graph.values])
        self.do_all(6)
        self.do_all(30)

    def do_all(self, n_stuff):
        trees = [Tree().generate_tree(3, None) for _ in range(n_stuff)]
        for t in trees:
            t.set_value_from_children('root')
        graphs = [TreeMobject(tree, node_size=0.3) for tree in trees]

        group = VGroup(*graphs)
        group.arrange_in_grid(buff=2)
        for g in graphs:
            g.update_labels()
        self.play(self.camera.auto_zoom(graphs, margin=2))
        self.play(Create(group))

        # labels = [Create(l) for graph in graphs for l in graph.values]
        # self.play(*labels)

        green_nodes = []
        blue_nodes = []
        for i, tree in enumerate(trees):
            green_nodes += [graphs[i][n[1]['data'].id]
                for n in tree.g.nodes(data=True) if n[1]['data'].turn]
            blue_nodes += [graphs[i][n[1]['data'].id]
                for n in tree.g.nodes(data=True) if not n[1]['data'].turn]

        # self.wait(duration=0.5)
        # self.play(*map(lambda x: Wiggle(x, scale_value=1.3), blue_nodes))
        # self.wait(duration=0.5)
        # self.play(*map(lambda x: Wiggle(x, scale_value=1.3), green_nodes))

        animations = {}
        def monte_carlo_callback(i):
            def re(node, edge, n):
                if n not in animations:
                    animations[n] ={"nodes": [], "edges": []}
                animations[n]["nodes"].append(graphs[i].animate_node(node.id).set_color(RED))
                if edge:
                    animations[n]["edges"].append(graphs[i].edges[edge].animate.set_color(RED))
            return re

        monte_carlo_results = []
        for i in range(len(graphs)):
            monte_carlo_results.append(trees[i].monte_carlo('root', callback=monte_carlo_callback(i), traversed=[]))
        for i in sorted(animations.keys()):
            self.play(*animations[i]["nodes"], run_time=0.5)
            if len(animations[i]["edges"]):
                self.play(*animations[i]["edges"], run_time=0.5)



        fallback_animations = [[] for _ in range(len(max(monte_carlo_results, key=len)) * 4)]
        for i, nodes in enumerate(monte_carlo_results):
            for k in range(len(nodes) - 1):
                n = k + 2
                decision = nodes[-n]
                out_edges : List[Tuple[str, str]] = list(trees[i].g.out_edges(decision.id))


                vertices = []
                edges = []
                vs = []

                for u, v in out_edges:
                    node_data = trees[i].node_data(v)
                    vertices += [v]
                    edges += [(u, v)]
                    vs += [node_data.value]


                fallback_animations[k * 4] += [Succession(*[Wiggle(graphs[i][v]) for _, v in out_edges])]

                vs=np.array(vs)
                s = vs.sum()
                vs /= vs.sum()
                probs = decision.probs
                for j in range(len(probs)):
                    probs[j] += vs[j] - decision.value / s
                    probs[j] = max(0, probs[j])
                probs /= probs.sum()


                edge_ani=[]
                for j, edge in enumerate(edges):
                    edge_ani += [graphs[i].edges[edge]
                        .animate.set_stroke_width(probs[j] * 15)]
                fallback_animations[k * 4 + 1] += edge_ani




        for animation in fallback_animations:
            if len(animation):
                self.play(*animation, run_time=0.5)
        self.play(*[FadeOut(g) for g in graphs])


