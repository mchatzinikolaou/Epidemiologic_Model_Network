import networkx as nx
import matplotlib.pyplot as plt
import Node
import numpy as np


# TODO
# Create node & adjacency files (check multi-graphs and shapefiles)
#xrisimopoiise numpy gia floating points

class PopulationNet(nx.DiGraph):
    """
    The network that models the communication between the nodes.
    """

    def addNode(self, Population, Name):
        """
        Adds a new node to the network.
        :param Population: The population of the node
        :param Name: The name of the node.
        """

        NewPopNode = Node.PopulationNode(Population, name=Name)
        self.add_node(NewPopNode)
        return NewPopNode


    def addEdge(self, NodeName1, NodeName2, Weight=1):
        """
        Adds a new edge to the graph. The edge initially represents the traffic between the two nodes.
        It could be chosen to represent other things, such as profit/trade flow.
        :param NodeName1: Departure node
        :param NodeName2: Arrival node
        :param Weight: The traffic (this can be expanded)
        :return:
        """
        if NodeName1 == NodeName2:
            print("Can't create self-loops. (", NodeName1, " to ", NodeName2, ")")
            return
        Node1 = self.getNodeByName(NodeName1)
        Node2 = self.getNodeByName(NodeName2)
        self.add_edge(Node1, Node2, weight=Weight)   #vale endexomenws type integer stin synartisi.

    # draw topology
    def draw(self):
        """
        Draws the graph
        """

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
        """
        Returns the node object using it's name.
        (The object is the entity itself, as opposed to a copy of it. This allows us to
        alter its variables such as the populations etc.)

        :param name: The searched name.
        :return: The node object.
        """
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
        """
        Returns all neighbours of the named node.
        This can be used to access the traffic between the nodes , alter it ,
        add attributes (more "weight variables") etc.

        :param name: The name of the node in question
        :return: neighbourList = All the neighbours
        """
        neighbourList = []
        for edge in self.getEdges():
            if edge[0].name == name:
                neighbourList.append(edge[1].getName())
        return neighbourList

    def isEmpty(self):
        return self.number_of_nodes() == 0


    def departures(self, days=1):
        """
        This IS the traffic between the nodes.
        For each node , get all its neighbours and send travellers according toa  predefined schedule.

        :param days: How many days we want to be ran. This should later change to a schedule, where for
        each day we retrieve schedule[day], which will be a list of weights for each node.
        """
        for day in range(1, days):
            for edge in self.edges.data():
                print("From ", edge[0].getName(), " to ", edge[1].getName(), ": ", edge[2]['weight'], " travelers")
                edge[0].TravelTo(edge[1], edge[2]['weight'])
            for node in self.nodes:
                node.advanceByDays(1)
            print(day)

    def plotNodeHistory(self, nodeName):

        """
        Plot the whole history of the "nodeName" named Node.

        :param nodeName: the name of the node
        """


        history = self.getNodeByName(nodeName).getHistory()

        plt.figure(nodeName)
        t = range(0, np.size(history, 0))

        S = []
        I = []
        R = []

        for i in range(0, len(history)):
            S.append((history[i])[0])
            I.append((history[i])[1])
            R.append((history[i])[2])

        plt.plot(t, S, 'r-')
        plt.plot(t, I, 'g-')
        plt.plot(t, R, 'b-')
        plt.show()

    def plotTotalHistory(self):
        """
        Plot the history of the whole system.

        This requires fetching the history of each node and summing their absolute values
        """
        history = []
        for node in self.nodes:
            if len(history) != 0:
                history = np.add(history, node.getHistory())
            else:
                history = node.getHistory()

        plt.figure("Total History")
        t = range(0, np.size(history, 0))

        S = []
        I = []
        R = []

        for i in range(0, len(history)):
            S.append((history[i])[0])
            I.append((history[i])[1])
            R.append((history[i])[2])

        plt.plot(t, S, 'r-')
        plt.plot(t, I, 'g-')
        plt.plot(t, R, 'b-')
        plt.show()



def Demonstrate():
    """
    Run a test by creating three nodes, for Athens, Chania and Rhodes.

    :return:
    """

    newnet = PopulationNet()


    #HERE WE SHOULD READ THE EDGELISTS ETC
    newnet.addNode(100000, "Athens")
    newnet.addNode(112312, "Rhodes")
    newnet.addNode(434123, "Chania")
    newnet.addEdge("Athens", "Rhodes", 500)
    newnet.addEdge("Rhodes", "Athens", 500)
    newnet.addEdge("Athens", "Chania", 50)
    newnet.addEdge("Rhodes", "Chania", 50)



    #INITIALIZE AND RUN FOR x days

    newnet.testInfectNode("Athens")
    #for i in range(1,9):
    #    newnet.departures(100)

    newnet.departures(900)


    newnet.draw()

    #PLOT RESULTS
    newnet.plotNodeHistory("Chania")
    newnet.plotNodeHistory("Athens")
    newnet.plotNodeHistory("Rhodes")

    newnet.plotTotalHistory()
    print("A-OK!")

Demonstrate()