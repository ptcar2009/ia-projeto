from manim import *
from typing import Union
import networkx as nx
class Node():
    def __init__(self, id: str,  turn: bool, svg):
        self.svg = svg
        self.turn = turn
        self.id = id

class Tree():
    def __init__(self) -> None:
        self.g = nx.DiGraph()

    def add_child(self, node: Node, parent: Union[Node, None]):
        self.g.add_node(node.id, data=node)
        if parent:
            self.g.add_edge(parent.id, node.id)


    def node_data(self, node) -> Node:
        return self.g.nodes(data=True)[node]['data']



class CircleSvg(VGroup):
    def __init__(self, radius, color, fill_opacity, svg, **kwargs):
        super().__init__()
        self.circle = Circle(radius=radius, color=color, fill_opacity=fill_opacity, stroke_color=WHITE, stroke_width=10, **kwargs)
        self.add(self.circle)

        self.svg = SVGMobject(file_name=svg, fill_opacity=0, stroke_color=WHITE, color=WHITE, fill_color=WHITE)

        self.svg.scale(1/(max(self.svg.height, self.svg.width + 0.3)))

        self.add(self.circle)
        self.add(self.svg)

class TreeMobject(Graph):
    def __init__(self,  tree: Tree, node_size=0.3) -> None:

        self.tree = tree
        self.node_size = node_size
        nodes = list(map(lambda x: x[1]["data"], tree.g.nodes.items()))
        edges = list(tree.g.edges)

        args = {
            "vertices": list(map(lambda x: x.id, nodes)),
            "vertex_config": self.node_to_style(nodes),
            "vertex_type": CircleSvg,
            "edges": edges,
            "edge_type": Line,
            "layout": "tree",
            "root_vertex": 'root',
            "layout_scale": (3, 1.8),
            "edge_config": {"stroke_width": 11}
        }
        super().__init__(**args)





    def node_to_style(self, nodes: List[Node]):

        return {node.id: {
                "radius": self.node_size,
                "color": BLUE if node.turn else GREEN,
                "fill_opacity": 1,
                "svg": node.svg,
                } for node in nodes}

    def animate_node(self, node):
        return self[node].animate


class PedraPapelOuTesoura(Scene):
    def __init__(self, renderer=None, always_update_mobjects=False, random_seed=None, skip_animations=False):
        def camera_class():
            return Camera(background_color="#1E1E1E")
        super().__init__(renderer, camera_class, always_update_mobjects, random_seed, skip_animations)
    def construct(self):
        g = VGroup()
        pedra_n = Node("pedra", True, "media/images/pedra.svg")
        papel_n = Node("papel", True, "media/images/papel.svg")
        tesoura_n = Node("tesoura", True, "media/images/tesoura.svg")
        root_n = Node("root", False, "media/images/qm.svg")

        tree = Tree()
        tree.add_child(root_n, None)
        tree.add_child(tesoura_n, root_n)
        tree.add_child(papel_n, root_n)
        tree.add_child(pedra_n, root_n)

        graph = TreeMobject(tree, 1)
        g.add(graph)

        # 2s
        self.play(Create(graph["root"]), run_time=2)

        self.wait(duration=17.8)

        #
        self.play(Create(graph["pedra"]),)
        self.play(Create(graph["papel"]))
        self.play(Create(graph["tesoura"]))

        self.wait()

        self.play(*[Create(l) for l in graph.edges.values()])

        self.wait(12)

        pedra_v = Text("0").next_to(graph['pedra'], DOWN)
        papel_v = Text("0").next_to(graph['papel'], DOWN)
        tesoura_v = Text("0").next_to(graph['tesoura'], DOWN)
        self.play(Write(pedra_v), Write(papel_v), Write(tesoura_v))

        self.wait(16.3)

        self.play(Wiggle(graph["pedra"], n_wiggles=5))
        self.play(Wiggle(graph["pedra"], n_wiggles=5))
        self.play(Wiggle(graph["pedra"], n_wiggles=5))


        self.wait(2)
        enemy_empty = Circle(color=RED,  fill_opacity=1, radius=1, stroke_width=10, stroke_color=WHITE)
        g.add(enemy_empty)

        self.play(Create(enemy_empty), g.animate.arrange(), FadeOut(pedra_v), FadeOut(papel_v), FadeOut(tesoura_v))
        enemy = CircleSvg(color=RED,  fill_opacity=1, radius=1, svg="media/images/papel.svg").move_to(enemy_empty)

        self.wait(4)
        self.play(Transform(enemy_empty, enemy))
        self.wait()
        self.play(Write(Text("1").next_to(enemy, DOWN)))

        self.wait()

