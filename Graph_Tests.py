import networkx as nx
import matplotlib.pyplot as plt
import Node

class NodeNet():

    #constructor
    def __init__(self):
        self.CityGraph=nx.Graph()

    #add a new node
    def addNode(self,Population,Name):
        NewNode=Node.PopulationNode(Population, name=Name)
        self.CityGraph.add_node(NewNode)

    #add a connection between two nodes.
    def addEdge(self,Node1,Node2):
        self.CityGraph.add_edge(Node1,Node2)

    #draw topology
    def draw(self):
        lablist= {}
        for node in self.getNodes():
            lablist[node]=node.name
        #rename labels for showing purposes
        H = nx.relabel_nodes(self.CityGraph,lablist)
        nx.draw(H, with_labels=True)
        plt.show()

    #get all nodes
    def getNodes(self):
        return self.CityGraph.nodes

    #get all edges
    def getEdges(self):
        return self.CityGraph.edges

    #return a node by name.
    def getNode(self,name):
        for node in self.getNodes():
            if node.name == name:
                return node

    #Test infect a node
    def infectNode(self,name):
        self.getNode(name).TestInfect()

    #Advance by some days each node.
    def advance(self,days=1):
        nodes=self.getNodes()
        for node in nodes:
            node.advanceByDays(days)

    #Find neighbours of given edge
    def FindNeighbours(self,name):
        neighbourList=[]
        for edge in self.getEdges():
            if edge[0].name == name:
                neighbourList.append(edge[1])
        return neighbourList



newnet = NodeNet()
newnet.addNode(5000000,"Athens")
newnet.addNode(100000,"Rhodes")
newnet.addNode(1000,"Tzimpron")

for node1 in newnet.getNodes():
    for node2 in newnet.getNodes():
        if node1 != node2 :
            newnet.addEdge(node1,node2)

print(newnet.FindNeighbours("Athens"))

newnet.infectNode("Athens")
newnet.advance(100)
newnet.draw()
