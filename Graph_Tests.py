import networkx as nx
import matplotlib.pyplot as plt
import Node


# TODO
# Create node & adjacency files (check multi-graphs and shapefiles)


class PopulationNet(nx.DiGraph):

    # add a new node
    def addNode(self, Population, Name):
        NewPopNode = Node.PopulationNode(Population, name=Name)
        self.add_node(NewPopNode)
        return NewPopNode

    # add a connection between two nodes.
    def addEdge(self, NodeName1, NodeName2, Weight=1):
        if NodeName1 == NodeName2:
            print("Can't create self-loops. (", NodeName1, " to ", NodeName2, ")")
            return
        Node1 = self.getNodeByName(NodeName1)
        Node2 = self.getNodeByName(NodeName2)
        self.add_edge(Node1, Node2, weight=Weight)

    # draw topology
    def draw(self):
        lablist = {}
        for node in self.getNodes():
            lablist[node] = node.name
        # rename labels for showing purposes
        H = nx.relabel_nodes(self, lablist)
        nx.draw(H, with_labels=True)
        nx.draw_networkx_edge_labels(H, pos=nx.spring_layout(H))
        plt.show()

    # get all nodes
    def getNodes(self):
        return self.nodes

    # return available node names.
    def getNodeNames(self):
        nodelist = []
        for node in self.nodes:
            nodelist.append(node.getName())
        return nodelist

    # get all edges
    def getEdges(self):
        return self.edges

    # return a node by name.
    def getNodeByName(self, name):
        for node in self.getNodes():
            if node.getName() == name:
                return node

    # Test infect a node
    def testInfectNode(self, name):
        self.getNodeByName(name).TestInfect()

    # Advance by some days each node.
    def advance(self, days=1):
        nodes = self.getNodes()
        for node in nodes:
            node.advanceByDays(days)

    # Find neighbours of given edge
    def FindNeighbours(self, name):
        neighbourList = []
        for edge in self.getEdges():
            if edge[0].name == name:
                neighbourList.append(edge[1].getName())
        return neighbourList

    def isEmpty(self):
        return self.number_of_nodes() == 0

    # update the departures for each node
    def departures(self):
        # for each connection
        for edge in self.edges.data():
            print("From ", edge[0].getName(), " to ", edge[1].getName(), " with weight ", edge[2]['weight'])


newnet = PopulationNet()
newnet.addNode(3000, "Athens")
newnet.addNode(4000, "Rhodes")
newnet.addNode(43000, "Chania")
newnet.addEdge("Athens", "Rhodes", 500)
newnet.addEdge("Rhodes", "Athens", 300)
newnet.addEdge("Athens", "Chania", 200)
newnet.addEdge("Rhodes", "Chania", 100)

newnet.departures()
newnet.draw()
