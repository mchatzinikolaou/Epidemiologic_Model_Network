import Graph as gt
from Graph import PopulationNet as pn
from random import random
import numpy as np
import scipy.optimize.nnls as nnls
import matplotlib.pyplot as plt

def singleRegression():
    [history, beta, gamma] = gt.runRandomNode(500)
    print(beta, gamma)

    params = gt.betaGammaFromEquations(history)

    print("Using equations : beta ", params[0], "gamma: ", params[1])
    print("Errors : ", abs(params[0] - beta) / beta, " ", abs(gamma - params[1]) / gamma)

def generateGraph(betas,gammas,p,num_nodes=20):

    InitialPopulations=1e04*np.ones(num_nodes)
    Populations = InitialPopulations

    network = pn()

    for i in range(0, num_nodes):
        network.addNode(Populations[i], str(i), betas[i], gammas[i])
        for j in range(0, i):

            if (random()>p):
                network.addEdge(str(i), str(j), 1)
            if (random() > p):
                network.addEdge(str(j), str(i), 1)
    return network

def betaGamas1(num_nodes):
    parameters = gt.GenerateParameters(num_nodes)
    truebetas = [i[0] for i in parameters]
    truegammas = [i[1] for i in parameters]
    return [truebetas,truegammas]

def betaGamas2(b_min, b_max, g_min, g_max, num_nodes):
    parameters = gt.GenerateRandomParameters(b_min, b_max, g_min, g_max, num_nodes)
    truebetas = [i[0] for i in parameters]
    truegammas = [i[1] for i in parameters]
    return [truebetas, truegammas]

def getHistories(network,DAYS,N):
    NodeHistories = []
    for nodename in network.getNodeNames():
        NodeHistories.append(network.getNodeByName(nodename).getHistory())
    History = network.getTotalHistory()

    Ss = []
    Is = []
    Rs = []
    Populations = []
    for k in range(0, N):
        nodeHistory = NodeHistories[k]
        nodePopulationHistory = network.getNodeByName(str(k)).PopulationHistory
        newS = []
        newI = []
        newR = []
        for day in range(0, DAYS - 1):
            newS.append(np.divide(nodeHistory[day][0], nodePopulationHistory[day]))
            newI.append(np.divide(nodeHistory[day][1], nodePopulationHistory[day]))
            newR.append(np.divide(nodeHistory[day][2], nodePopulationHistory[day]))
        Ss.append(newS)
        Is.append(newI)
        Rs.append(newR)
        Populations.append(nodePopulationHistory)
    return  [History,Ss,Is,Rs,Populations]

def getDeltasAndMatrix(Ss,Populations,DAYS,N):
    DeltaS = []
    MeasurementMatrix = []
    for i in range(1, DAYS - 1):
        DeltaS.append(S[i] - S[i - 1])
        innerList = []
        for k in range(0, N):
            innerList.append((-np.multiply(Ss[k][i - 1], Populations[k][i - 1]) * Is[k][i - 1]))
        MeasurementMatrix.append(innerList)

    return MeasurementMatrix,DeltaS

DAYS=500
N=20
truebetas,truegammas=betaGamas1(N)

repeats=N*N
predicted_b=[]
for repeat in range(0,repeats):

    network=generateGraph(truebetas,truegammas,p=(2 / (N - 1)),num_nodes=N)
    network.testInfect()
    network.departures(DAYS)

    [History,Ss,Is,Rs,Populations]=getHistories(network,DAYS,N)

    S=History[0]
    I=History[1]
    R=History[2]

    [MeasurementMatrix, DeltaS]=  getDeltasAndMatrix(Ss,Populations,DAYS,N)

    MeasurementMatrix = np.array(MeasurementMatrix)
    DeltaS = np.array(DeltaS)


    betas=nnls.nnls(MeasurementMatrix,DeltaS,maxiter=10*len(MeasurementMatrix))[0]


    errors=100*np.divide(np.subtract(betas,truebetas),truebetas)
    idx=np.argmax(errors)

    #testBetas=np.matmul(np.linalg.pinv(MeasurementMatrix),DeltaS)

    predicted_b.append(list(betas))
    print("repeat ",str(repeat+1), " just finished")
    #network.plotTotalHistory()

mean_b=np.mean(predicted_b, axis=0)
print(truebetas)
print(mean_b)

plt.scatter(truebetas,mean_b)
plt.title("Scatter plot of real beta values and predicted beta values")
plt.ylabel("Predicted beta(mean)")
plt.xlabel("True beta")
plt.show()

median_b=np.median(predicted_b,axis=0)

plt.scatter(truebetas,median_b)
plt.title("Scatter plot of real beta values and predicted beta values")
plt.ylabel("Predicted beta(median)")
plt.xlabel("True beta")
plt.show()

print("True betas " ,truebetas)
print("Mean betas : ",mean_b)
print("Median betas: ",median_b)

print("Mean absolute error  : ",np.mean(np.subtract(truebetas,mean_b)))
print("Median absolute error  : ",np.mean(np.subtract(truebetas,median_b)))