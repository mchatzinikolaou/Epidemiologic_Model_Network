import networkx as nx
import matplotlib.pyplot as plt
import Node
import numpy as np
import random
import InfectionTree as Itree

class PopulationNet(nx.DiGraph):
    """
    The network that models the communication between the nodes.
    """
    InfTree = Itree.InfectionTree()

    def addNode(self, Population, Name,beta=0.5,gamma=0.05):
        """
        Adds a new node to the network.
        :param Population: The population of the node
        :param Name: The name of the node.
        """
        if beta==None and gamma==None:
            NewPopNode = Node.PopulationNode(Population, Name)
        else:
            NewPopNode=Node.PopulationNode(Population,Name,beta,gamma)


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

    def getNodes(self):
        return self.nodes

    def getNodeNames(self):
        nodelist = []
        for node in self.nodes:
            nodelist.append(node.getName())
        return nodelist

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

    def advance(self, days=1):
        nodes = self.getNodes()
        for node in nodes:
            node.advanceByDays(days)

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
            # TODO

            for edge in self.edges.data():
                # print("From ", edge[0].getName(), " to ", edge[1].getName(), ": ", edge[2]['weight'], " travelers")
                newInfection = edge[0].TravelTo(edge[1], edge[2]['weight'])
                if newInfection:
                    print("Node ", edge[0].name, "infected node ", edge[1].name," on day ",day)
                    self.infectionEvents.append([day, edge[0], edge[1]])
                    self.InfTree.AddEdgeEvent(day, [edge[0].name, edge[1].name])
            # Print (newly) infected nodes
            for node in self.nodes:
                node.updatePopulation()
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

def GenerateNodes(TotalPopulation, num_nodes, node_parameters, PopulationDistributionType="uniform"):
    nodes = []
    if PopulationDistributionType == "uniform":
        for i in range(0, num_nodes):
            nodes.append(Node.PopulationNode(TotalPopulation / num_nodes, str(i), node_parameters[i][0], node_parameters[i][1]))
    return nodes

def GenerateSignleNode(Population, node_parameters):
    return Node.PopulationNode(Population,name=str(random.random()), beta=node_parameters[0], gamma=node_parameters[1])

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
        # newnet.addNode((G.degree[i] / G.number_of_edges()) * TotalPopulation, str(i))
        newnet.addNode(int(TotalPopulation / numberOfNodes), str(i))

    for edge in G.edges:
        newnet.add_edge(list(newnet.nodes)[edge[0]], list(newnet.nodes)[edge[1]], weight=50)

    return newnet

def plotHistory(History):
    S = []
    I = []
    R = []
    for i in History:
        S.append(i[0])
        I.append(i[1])
        R.append(i[2])

    t = range(0, np.size(S, 0))
    fig = plt.figure()
    plt.plot(t, S, 'r-', t, I, 'g-', t, R, 'b-')
    plt.legend(("Susceptible", "Infected", "Removed"))
    # fig.savefig(name)
    plt.show()

def runSimulation(totalNodes=50, totalPopulation=1e08, degree=1, days=500):
    if degree < 1:
        print("subcritical region")
    elif degree == 1:
        print("critical region")
    elif degree <= np.log(totalNodes):
        print("supercritical region")
    else:
        print("Connected region")
    if totalNodes > 1:
        net = CreateRandomNetwork(totalPopulation, p=degree * (2 / (totalNodes - 1)), numberOfNodes=totalNodes)
        # net=CreateNetwork_sf(totalPopulation,totalNodes)
        net.testInfect()
        net.departures(days)
        # net.plotTotalHistory()
        #net.InfTree.show()
        return net.getTotalHistory(), net.infectionEvents  # if NaN events, can't return

def runAndPlot(nodes=100, TotalPopulation=1e07, p=1.2, days=500,N=5,show_lines=False):
    # Run n simulations
    [History, results] = runSimulation(nodes, TotalPopulation, p, days)
    for i in range(1, N):
        [newHistory, newResults] = runSimulation(nodes, TotalPopulation, p, days)
        History[0] = np.add(History[0], newHistory[0])
        History[1] = np.add(History[1], newHistory[1])
        History[2] = np.add(History[2], newHistory[2])

    S = np.divide(History[0], N)
    I = np.divide(History[1], N)
    R = np.divide(History[2], N)

    t = range(0, np.size(S, 0))

    days = list(np.array(results)[:, 0])

    plt.figure()
    plt.plot(t, S, 'g-', t, I, 'r-', t, R, 'b-')
    plt.title('Progress of the epidemic for '+str(nodes)+' nodes (mean degree='+str(p)+")")
    plt.ylabel('Population')
    plt.xlabel('Day')
    # Visualization of infection events
    if show_lines == True:
        i = 0
        while i < len(days):
            prev_day = days[i]
            length = 1
            j = 0
            while i + j < len(days) and days[i + j] == prev_day:
                length = length + 1
                j = j + 1
            plt.vlines(days[i], 0, (length - 1) * TotalPopulation/nodes)
            i = i + j

    plt.legend(("Susceptible", "Infected", "Removed"))
    # fig.savefig(name)
    plt.show()

def GenerateRandomParameters(b_min, b_max, g_min, g_max, num_nodes):
    return np.random.uniform([b_min, g_min], [b_max, g_max], (num_nodes, 2))

def GenerateLinearParameters(b_min, b_max, g_min, g_max,size=20):
    ParamList=[]
    for beta in np.linspace(b_min,b_max,size):
        for gamma in np.linspace(g_min, g_max, size):
            ParamList.append([beta,gamma])
    return ParamList

def GenerateParameters(num_nodes=100):
    b_range = [0.1, 0.5]
    g_range = [0.05, 0.15]
    return GenerateRandomParameters(b_range[0], b_range[1], g_range[0], g_range[1], num_nodes)

def runRandomNode(days=1000,population=1e07):
    nodeParameters = GenerateParameters(1)
    nodes=GenerateNodes(population, 1, nodeParameters)[0]
    nodes.TestInfect()
    history = nodes.advanceByDays(days)
    return history,nodes.b,nodes.g

def discreteDerivative(Points):

    for i in range(len(Points)-1):
        Points[i][:]=np.subtract(Points[i+1][:],Points[i][:])
    return Points[0:-1][:]

def betaGammaFromEquations(History):

    Total_Population=sum(History[0])

    beta=0
    gamma=0
    t=0

    betas=[]
    gammas=[]

    for t in range(len(History)-1):
        #MEDIAN INSTEAD
        if History[t+1][1]-History[t][1]==0 or History[t+1][0]-History[t][0]==0:
            break
        St = np.divide(History[t+1][0], Total_Population)
        St_1 = np.divide(History[t][0], Total_Population)
        It = np.divide(History[t+1][1], Total_Population)
        It_1 = np.divide(History[t][1], Total_Population)

        d_beta= - np.divide((St - St_1), (np.multiply(St_1, It_1)))

        d_gamma= 1 - np.divide(It, It_1) + np.multiply(d_beta, St_1)
        #print(St - St_1,It-It_1)
        beta =  np.add(beta,d_beta)
        gamma = np.add(gamma,d_gamma)

        #REMOVE
        betas.append(d_beta)
        gammas.append(d_gamma)
        #REMOVE

    beta=np.median(betas)
    gamma=np.median(gammas)


    #REMOVE
    plt.plot(range(0,len(betas)),betas,gammas)
    plt.show()
    #REMOVE
    return [beta,gamma]



runAndPlot(nodes=100, TotalPopulation=1e07, p=49, days=300,N=5,show_lines=True)
