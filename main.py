import Graph_Tests as nets
import time


#Menu Options.
options=[
    '1.Add new Node',
    '2.Add edge',
    '3.Print Network and exit',
    '4.Load Network',
    '5.Exit'
]


#Menu Handler
def printMenu():
    for option in options:
        print(option)
    return int(input())

#Adds a new node by it's name and it's population
def addNewNode(Network):

    #add the node
    name = input("Enter node name: ")
    pop = input(("Enter node population: "))
    Network.addNode(pop, name)


def addNewEdge(Network):

    #Print available nodes
    if(Network.number_of_nodes()<2):
        print("Not enough nodes available")
        return



    #select node1
    print()



    selection=-1
    i=0
    Node1=[]
    Node2=[]
    nodelist = Network.getNodes()
    while(selection<0 or selection>len(nodelist)):
        for node in nodelist:
            print(str(i+1)+". "+node.getName())
            i=i+1
        selection=int(input("Select node 1 : "))
        if not (selection<0 or selection>len(nodelist)):
            Node1=nodelist[selection-1]
        else:
            print("Wrong option, chose again.")
    nodelist.remove(Node1)

    selection = -1
    while (selection < 0 or selection > len(nodelist)):
        for node in nodelist:
            print(str(i + 1) + ". " + node.getName())
            i = i + 1
        selection1 = int(input("Select node 1 : "))
        if not (selection1 < 0 or selection1 > len(nodelist)):
            Node2 = nodelist[selection-1]
        else:
            print("Wrong option, chose again.")

    #select weight
    #make connection



#Boot

#Create a new Network.
network=nets.PopulationNet()


#main program.
while (True):
    #Menu and options
    option=printMenu()
    if(option==1):
        addNewNode(network)
    if (option == 2):
        addNewEdge(network)
    if(option==3):
        network.draw()
    if(option==4):
        continue
    if(option==5):
        exit(0)




"""
def addEdge(Network):
    print("list of nodes: " + Network.getNodeNames())
    #Select neighbours to be added.
    selection=None
    while ((selection != "n") and (selection != "N")):
        print("Add neighbouring nodes from (n to skip): ")
        i=0
        nodes=Network.getNodes()
        selections={i in range(1,len(nodes)):nodes}
        print(selections)
        for neighbour in nodes:
            print(str(i)+neighbour.getName())
            i+=1
        selection=input()



        #Add neighbour and it's weight
        if((selection != "n") and (selection != "N") and (selection in nodes)):
            weight=-1
            while(weight<0 or weight>1):
                weight= input("Give edge weight: ")
            Network.addEdge(Node,selection,weight)
"""