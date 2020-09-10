import numpy as np

class Node_Network:

    #Initialize two forms, one with the adjascency list and one with the matrix
    def __init__(self,numOfCities):
        self.AdjacencyMatrix=np.zeros([numOfCities,numOfCities])
        self.AdjacencyList=[]
        for node in range(1,numOfCities):
            self.AdjacencyList.append([])


    #What kind of file are we reading? Parse and update map
    def readTopology(self):
        return

    #Add one node to the lists and update the matrix.
    def AddNode(self):
        self.AdjacencyMatrix=np.resize(self.AdjacencyMatrix,(np.size(self.AdjacencyMatrix,0)+1,np.size(self.AdjacencyMatrix,1)+1))
        print(np.size(self.AdjacencyMatrix, 0), np.size(self.AdjacencyMatrix, 1))
