import networkx as nx
import matplotlib.pyplot as plt
import Node
import numpy as np
import random
import InfectionTree as Itree

# TODO
# Create node & adjacency files (check multi-graphs and shapefiles)
# MC simulation of SIR

class PopulationNet(nx.DiGraph):
    """
    The network that models the communication between the nodes.
    """
    InfTree=Itree.InfectionTree()

    def addNode(self, Population, Name):
        """
        Adds a new node to the network.
        :param Population: The population of the node
        :param Name: The name of the node.
        """

        NewPopNode = Node.PopulationNode(Population, name=Name)
        self.infectionEvents = []
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
        self.add_edge(Node1, Node2, weight=Weight)  # vale endexomenws type integer stin synartisi.

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
        nx.draw_networkx(H, pos=nx.circular_layout(H))
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

    def getNodeDegree(self, name):
        return self.degree[self.getNodeIndexByName(name)]

    def getNodeIndexByName(self, name):
        i = 0
        for node in self.getNodes():
            if node.getName() == name:
                return i
            i = i + 1

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

    def testInfect(self):
        list(self.nodes)[0].TestInfect()

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
            print("Day " + str(day))
            for edge in self.edges.data():
                # print("From ", edge[0].getName(), " to ", edge[1].getName(), ": ", edge[2]['weight'], " travelers")
                newInfection = edge[0].TravelTo(edge[1], edge[2]['weight'])
                if newInfection:
                    print("Node ", edge[0].name, "infected node ", edge[1].name)
                    self.infectionEvents.append([day, edge[0], edge[1]])
                    # TODO
                    self.InfTree.AddEdgeEvent(day, [edge[0].name, edge[1].name])

            # TODO
            # Print (newly) infected nodes
            for node in self.nodes:
                node.advanceByDays(1)

    def depart_and_randomize_travel(self, p, days=1):
        for day in range(1, days):
            self.departures(1)
            self.ChangeConnections(p)

    def ChangeConnections(self, p):
        self.remove_edges_from(list(self.edges))
        for node1 in self.getNodeNames():
            for node2 in self.getNodeNames():

                if node1 != node2 and random.random() <= p:
                    self.addEdge(node1, node2, 50)

                # if node1 != node2 and random.random() <= p*p:
                #    self.addEdge(node1, node2, 50)
                #    self.addEdge(node2, node1, 50)

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

        plt.plot(t, S, 'r-', t, I, 'g-', t, R, 'b-')

        plt.show()

    def getTotalHistory(self):
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

        S = []
        I = []
        R = []

        for i in range(0, len(history)):
            S.append((history[i])[0])
            I.append((history[i])[1])
            R.append((history[i])[2])

        return [S, I, R]

    def plotTotalHistory(self):
        [S, I, R] = self.getTotalHistory()
        t = range(0, np.size(S, 0))
        name = "nodes_" + str(self.number_of_nodes()) + "_edges_" + str(self.number_of_edges()) + ".png"
        print(name)
        fig = plt.figure()
        plt.plot(t, S, 'r-', t, I, 'g-', t, R, 'b-')
        plt.legend(("Susceptible", "Infected", "Removed"))
        # fig.savefig(name)
        plt.show()


def CreateRandomNetwork(TotalPopulation, p, numberOfNodes=1):
    if p > 1 or p < 0:
        print("Invalid probability value")
        return

    newnet = PopulationNet()

    for i in range(0, numberOfNodes):
        newnet.addNode(int(TotalPopulation / numberOfNodes), str(i))

    for node1 in newnet.getNodeNames():
        for node2 in newnet.getNodeNames():
            if node1 != node2 and random.random() <= p:
                newnet.addEdge(node1, node2, 50)
    return newnet


def CreateNetwork_sf(TotalPopulation, numberOfNodes=1):
    newnet = PopulationNet()

    G = nx.scale_free_graph(numberOfNodes)
    for i in range(0, G.number_of_nodes()):
        #newnet.addNode((G.degree[i] / G.number_of_edges()) * TotalPopulation, str(i))
        newnet.addNode(int(TotalPopulation / numberOfNodes), str(i))

    for edge in G.edges:
        newnet.add_edge(list(newnet.nodes)[edge[0]], list(newnet.nodes)[edge[1]], weight=50)

    return newnet


def plotHistory(History):
    S = History[0]
    I = History[1]
    R = History[2]
    t = range(0, np.size(S, 0))
    fig = plt.figure()
    plt.plot(t, S, 'r-', t, I, 'g-', t, R, 'b-')
    plt.legend(("Susceptible", "Infected", "Removed"))
    # fig.savefig(name)
    plt.show()


def runSimulation(totalNodes, totalPopulation, degree, days):
    if degree < 1:
        print("subcritical regime")
    elif degree == 1:
        print("critical regime")
    elif degree <= np.log(totalNodes):
        print("supercritical regime")
    else:
        print("Connected regime")
    if totalNodes > 1:
        net = CreateRandomNetwork(totalPopulation, p=degree * (2 / (totalNodes - 1)), numberOfNodes=totalNodes)
        #net=CreateNetwork_sf(totalPopulation,totalNodes)
        net.testInfect()
        net.departures(days)
        # net.plotTotalHistory()
        net.InfTree.show()
        return net.getTotalHistory(), net.infectionEvents  # if NaN events, can't return


def runAndPlot():
    # Run n simulations
    N = 1
    [History, results] = runSimulation(50, 10e08, 1.2, 1000)
    for i in range(1, N):
        [newHistory, newResults] = runSimulation(3, 10e08, 1.2, 300)
        History[0] = np.add(History[0], newHistory[0])
        History[1] = np.add(History[1], newHistory[1])
        History[2] = np.add(History[2], newHistory[2])
        plotHistory(newHistory)
        plotHistory(History)

    S = np.divide(History[0], N)
    I = np.divide(History[1], N)
    R = np.divide(History[2], N)

    t = range(0, np.size(S, 0))

    days = list(np.array(results)[:, 0])

    plt.figure()
    plt.plot(t, S, 'g-', t, I, 'r-', t, R, 'b-')

    # Visualization of infection events
    i = 0
    while i < len(days):
        prev_day = days[i]
        length = 1
        j = 0
        while i + j < len(days) and days[i + j] == prev_day:
            length = length + 1
            j = j + 1
        plt.vlines(days[i], 0, (length - 1) * 1e08)
        i = i + j

    plt.legend(("Susceptible", "Infected", "Removed"))
    # fig.savefig(name)
    plt.show()


runAndPlot()
