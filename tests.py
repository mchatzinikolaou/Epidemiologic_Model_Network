import Graph_Tests as gt
from Graph_Tests import PopulationNet as pn
import numpy as np

#Το σφαλμα που υπαρχει ειναι λογω της στρογγυλοποιησης και οχι της μηχανης,
#καθως τα SIR στρογγυλοποιουνται στον πλησιέστερο ακέραιο.



def singleRegression():
    [history,beta,gamma] = gt.runRandomNode(500)
    print(beta,gamma)

    params=gt.betaGammaFromEquations(history)

    print("Using equations : beta ",params[0],"gamma: ",params[1])
    print ("Errors : ",abs(params[0]-beta)/beta," ",abs(gamma-params[1])/gamma)

[beta0,gamma0,Population0]=[1.2,0.2,1.1e05]
[beta1,gamma1,Population1]=[0.5,0.5,1e05]


network=pn()

network.addNode(Population0,"A",beta0,gamma0)
network.addNode(Population1,"B",beta1,gamma1)
network.addEdge("A","B",100)
network.addEdge("B","A",100)

network.testInfect()
network.departures(500)
node1=network.getNodeByName("A")
node2=network.getNodeByName("B")


network.plotTotalHistory()

History=network.getTotalHistory()
History1=node1.getHistory()
History2=node2.getHistory()


S=History[0]
I=History[1]
R=History[2]

S1=np.divide([i[0] for i in History1],Population0)
I1=np.divide([i[1] for i in History1],Population0)
R1=np.divide([i[2] for i in History1],Population0)

S2=np.divide([i[0] for i in History2],Population1)
I2=np.divide([i[1] for i in History2],Population1)
R2=np.divide([i[2] for i in History2],Population1)


print(History)
print(History1)
print(History2)
#we need M times these measurements :
# S1t
# S1(t-1)
# S2t
# S2(t-1)
# I2t
# I2(t-1)
# Population1
# Population2

DeltaS=[]
S1m_1=[]
S2m_1=[]
I1m_1=[]
I2m_1=[]
MeasurementMatrix=[]
for m in range(10,30,10):
    DeltaS.append(S[m]-S[m-1])
    MeasurementMatrix.append([  -S1[m-1]*I1[m-1]*Population0, -S2[m-1]*I2[m-1]*Population1    ])

MeasurementMatrix = np.array(MeasurementMatrix)
DeltaS=np.array(DeltaS)

print("Elements")
print(DeltaS)
print(MeasurementMatrix)
betas=np.linalg.solve(MeasurementMatrix, DeltaS)
print(betas)