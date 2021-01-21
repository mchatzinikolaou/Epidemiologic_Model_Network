import networkx as nx
import queue as q
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

class InfectionTree():

    # Trees : A dictionary {day: tree}
    edgeQueue= q.Queue()
    Tree=nx.DiGraph()

    #Add node to waiting queue
    def AddEdgeEvent(self,day,edge):
        self.edgeQueue.put([day,edge])

    #Update the Tree
    def updateTree(self):
        while not self.edgeQueue.empty():
            data=self.edgeQueue.get()
            edge=data[1]
            self.Tree.add_edge(edge[0],edge[1],day=str(data[0]))

    def show(self):
        self.updateTree()

        pos = graphviz_layout(self.Tree, prog="dot")
        nx.draw(self.Tree, pos,with_labels=True)
        labels = nx.get_edge_attributes(self.Tree, 'day')
        nx.draw_networkx_edge_labels(self.Tree, pos, edge_labels=labels)
        plt.show()

    #Save tree
        #Save the tree as a figure

    #Create gif
        #Create gif from figures
            #Probably each frame lasts for a while (fraction of set time or set time itself)
