import networkx as nx
import queue as q
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


class InfectionTree:
    edgeQueue = q.Queue()
    Tree = nx.DiGraph()

    def AddEdgeEvent(self, day, edge):
        self.edgeQueue.put([day, edge])

    def updateTree(self):
        while not self.edgeQueue.empty():
            data = self.edgeQueue.get()
            edge = data[1]
            self.Tree.add_edge(edge[0], edge[1], day=str(data[0]))

    def show(self):
        self.updateTree()

        pos = graphviz_layout(self.Tree, prog="dot")
        nx.draw(self.Tree, pos, with_labels=True)
        labels = nx.get_edge_attributes(self.Tree, 'day')
        nx.draw_networkx_edge_labels(self.Tree, pos, edge_labels=labels)
        plt.show()
