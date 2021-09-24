from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d

# DONE
# Basic SIR in single Population
# Migration between populations
# Calculate traveling population categories (multinomial distribution in "travel packets")

# TODO
#Make beta noise series of time.



# Regression
# Train a neural net

class PopulationNode:
    """
    Population Node.
    This class represents the progress in an isolated population has all the functionality to model the progress within
    the population , as well as the traffic of this population to and from others.

    """

    def __init__(self, total_Population, name, beta=0.2, gamma=0.05):
        """
        Initialize the node.
        This is used to create a new node with some name and parameters beta and gamma (for the SIR model, this will
        be expanded and eventually generalized to create any model.)


        :param total_Population: Initial population.
        :param name: This is the identifier of the node. Can be a real name or some artificial ID.
        :param beta: beta parameter of the SIR model
        :param gamma: gamma parameter of the SIR model

        We also initialize a list "history" which is used to story the progress so far.
        """

        # Demographics
        self.Population = total_Population
        self.S = 1.0
        self.I = 0.0
        self.R = 0.0

        self.b = beta
        self.g = gamma
        self.name = name
        self.history = []
        self.PopulationHistory = []

    def biasedAdvanceByDays(self, S_bias, I_bias, R_bias, days=1):
        """
                Progress the epidemic by specified amount of days (defaults to 1 day per step).

                Changes S ,I , R and the rest of the parameters according to the specified model.
                (This should be , again , generalized)

                :param days: the amount of days we want to progress the solution.
                :return: the progress through these days.
                """
        z = []
        S_old = self.S
        I_old = self.I
        R_old = self.R
        for day in range(0, days):
            # progress according to the model
            self.S += -np.multiply(np.multiply(self.b , S_old) , I_old)
            self.I += np.multiply(np.multiply(self.b , S_old) , I_old) - np.multiply(self.g , I_old)
            self.R += np.multiply(self.g , I_old)



            self.TravelFrom(S_bias, I_bias, R_bias)
            # update history.
            self.S = max(self.S, 0)
            z.append([np.multiply(self.S , self.Population), np.multiply(self.I , self.Population), np.multiply(self.R , self.Population)])

            # renew percentages for the next day.
            S_old = self.S
            I_old = self.I
            R_old = self.R

        self.history = self.history + z
        [self.S, self.I, self.R] = self.normalize([self.S, self.I, self.R])
        return z

    def advanceByDays(self, days=1):
        """
        Progress the epidemic by specified amount of days (defaults to 1 day per step).

        Changes S ,I , R and the rest of the parameters according to the specified model.
        (This should be , again , generalized)

        :param days: the amount of days we want to progress the solution.
        :return: the progress throughout these days.
        """
        z = []
        S_old = self.S
        I_old = self.I
        R_old = self.R
        for day in range(0, days):
            # progress according to the model
            self.S += -self.b * S_old * I_old
            self.I += self.b * S_old * I_old - self.g * I_old
            self.R += self.g * I_old

            # update history.
            self.S = max(self.S, 0)
            z.append([int(self.S * self.Population + 0.5), int(self.I * self.Population + 0.5),
                      int(self.R * self.Population + 0.5)])

            # renew percentages for the next day.
            S_old = self.S
            I_old = self.I
            R_old = self.R

        self.history = self.history + z

        [self.S, self.I, self.R] = self.normalize([self.S, self.I, self.R])
        return z

    def plotNodeHistory(self):

        """
        Plot the whole history of the "nodeName" named Node.

        :param nodeName: the name of the node

        """

        history = self.getHistory()
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

    def TravelTo(self, targetNode, groupSize):
        """
        :param targetNode: The node that will receive the travelers.
        :param groupSize: the total size of the travelling population
        """
        if groupSize >= self.Population:
            print("Not enough population in node "+str(self.name)+"!")
            return

        [self.S, self.I, self.R] = self.normalize([self.S, self.I, self.R])

        # Create passenger batch
        travelers = random.multinomial(n=groupSize, pvals=[self.S, self.I, self.R])

        # infect the other node.
        dS = travelers[0]
        dI = travelers[1]
        dR = travelers[2]

        #inform if there was an infection
        infected = targetNode.TravelFrom(dS, dI, dR)

        # remove passenger population from this node.
        # calculate using absolute values
        S_total = round(np.multiply(self.S , self.Population) - dS)
        if S_total < 0:
            S_total = 0
        I_total = round(np.multiply(self.I ,self.Population) - dI)
        if I_total < 0:
            I_total = 0
        R_total = round(np.multiply(self.R , self.Population) - dR)
        if R_total < 0:
            R_total = 0
        self.Population -= (dS + dI + dR)
        # update percentages.
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])
        return infected

    def TravelFrom(self, dS, dI, dR):
        """
        Receive the population , seperated into groups of Infected, Susceptible and Resolved(immune)
        people.
        :param dS: Susceptible incoming people.
        :param dI: Infected
        :param dR: Immune
        """
        infected = False
        if (self.I == 0) and dI >= 1:
            infected = True

        # Add absolute values
        S_total = round(np.multiply(self.S , self.Population) + dS)

        I_total = round(np.multiply(self.I ,self.Population) + dI)

        R_total = round(np.multiply(self.R , self.Population) + dR)
        self.Population += (dS + dI + dR)

        # recalculate percentages
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])
        return infected

    def TestInfect(self, dI=100):
        """
        "Inject" the population with dI infected individuals. This is used to initiate
        the epidemic.

        :param dI:
        """
        S_total = round(np.multiply(self.S , self.Population)) + 0
        I_total = round(np.multiply(self.I , self.Population)) + dI
        R_total = round(np.multiply(self.R , self.Population)) + 0
        self.Population += dI

        # recalculate percentages
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])

    def getName(self):
        return self.name

    def getHistory(self):
        return self.history

    def getTruePopulations(self):
        return [round(np.multiply(self.S , self.Population)), round(np.multiply(self.I , self.Population)),
                round(np.multiply(self.R , self.Population))]

    def updatePopulation(self):
        self.PopulationHistory.append(self.Population)

    def normalize(self, IntList):

        normalizedSIR = np.array(IntList)
        size = len(normalizedSIR)
        normalizedSIR = np.divide(normalizedSIR , normalizedSIR.sum())
        normalizedSIR = np.maximum(normalizedSIR, np.zeros(size))
        return normalizedSIR

    def assertCorrectPopulations(self):
        print(self.I + self.S + self.R)
        assert self.I + self.S + self.R == 1, "Population quotients don't sum up to 1"


def betaGammaFromEquations(History):

    betas = []
    gammas = []

    for t in range(1,len(History) - 1):

        if History[t + 1][1] - History[t][1] == 0 or History[t + 1][0] - History[t][0] == 0:
            continue
        Total_Population = sum(History[t])
        St = np.divide(History[t + 1][0], Total_Population)
        St_1 = np.divide(History[t][0], Total_Population)
        It = np.divide(History[t + 1][1], Total_Population)
        It_1 = np.divide(History[t][1], Total_Population)   

        d_beta = - np.divide((St - St_1), (np.multiply(St_1, It_1)))
        d_gamma = 1 - np.divide(It, It_1) + np.multiply(d_beta, St_1)

        betas.append(d_beta)
        gammas.append(d_gamma)

    beta = np.median(betas)
    gamma = np.median(gammas)

    return [beta, gamma]

def calculateBetaBiased(history, bS):
    S = [i[0] for i in history]
    I = [i[1] for i in history]
    beta_calculated = []

    for t in range(1, len(S)):
        population = sum(history[t])
        # The increased S without the bias, the bias is added afterwards
        St_init = S[t] - bS
        St_1 = S[t - 1]
        It_1 = I[t - 1]
        denom = np.multiply(St_1, It_1)
        beta_calculated.append(np.multiply(np.divide(bS, denom * population) - np.divide(St_init - St_1, denom), population))
    return np.median(beta_calculated)

def addNormalNoise(history,sigma=1):
    S=[i[0] for i in history]
    I=[i[1] for i in history]
    R=[i[2] for i in history]
    [Sn,In,Rn]=[[],[],[]]

    for i in range(0,len(S)):
        Sn.append( np.add(S[i],np.random.normal(loc=0, scale=sigma*S[i])))
        In.append( np.add(I[i],np.random.normal(loc=0, scale=sigma*I[i])))
        Rn.append( np.add(R[i],np.random.normal(loc=0, scale=sigma*R[i])))
    return [Sn,In,Rn]


def addMeasurementNoise(history,l=1):
    S = [i[0] for i in history]
    I = [i[1] for i in history]
    R = [i[2] for i in history]
    Population=[sum(i) for i in history]
    [Sn, In, Rn] = [[], [], []]
    for i in range(0, len(S)):
        Inoise= np.subtract(I[i], np.random.exponential(scale=1/l * I[i]))
        #Rnoise=np.subtract(R[i], np.random.exponential(scale=1/l * R[i]))
        Rnoise=R[i]
        In.append(max(Inoise,0))
        Rn.append(max(Rnoise,0))
        Sn.append(max(np.subtract(Population[i],np.add(Rnoise,Inoise)),0))

    return [Sn, In, Rn]

def ExponentialNoiseTest(truebeta=0.2,max_lambda=200,min_lambda=1,points=80,MAX_REPEATS=50):
    testNode = PopulationNode(3875000, name="Athens", beta=truebeta)

    testNode.TestInfect(10)
    testNode.advanceByDays(1500)

    print("True beta: ", testNode.b)
    history = testNode.getHistory()

    # calculate betas with noise
    betas = []
    lambdas = np.linspace(min_lambda,max_lambda,points)
    for l in lambdas:
        iter_betas = []
        #for each sigma
        Ss=[]
        Is=[]
        Rs=[]
        for repeat in range(0,MAX_REPEATS):
            noiseHistory = addMeasurementNoise(history, l=l)

            S = noiseHistory[0][:]
            I = noiseHistory[1][:]
            R = noiseHistory[2][:]

            #take rolling averages

            M=60
            S = uniform_filter1d(S, size=M)
            I = uniform_filter1d(I, size=M)
            R = uniform_filter1d(R, size=M)

            Ss.append(S)
            Is.append(I)
            Rs.append(R)


            noiseHistory = [[S[i], I[i], R[i]] for i in range(0, len(S))]
            iter_betas.append(betaGammaFromEquations(noiseHistory)[0])
            if repeat%10 ==0  :
                print(l,repeat)


        Ss=np.mean(Ss,axis=0)
        Is=np.mean(Is,axis=0)
        Rs=np.mean(Rs,axis=0)


        iter_betas=np.sort(iter_betas)[int(MAX_REPEATS/3):2*int(MAX_REPEATS/3)]
        print("beta: ",np.median(iter_betas))
        betas.append(np.median(iter_betas))
        #plt.plot(range(0,len(iter_betas)),iter_betas)
        #plt.show()
        print("done lambda",l)

    error =np.abs( np.divide(np.subtract(betas, truebeta), truebeta))
    return [error, lambdas]


def RunNoiseTests():

    R0 = 1
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(sigmas, error)
    axs[0, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[0, 0].grid(axis='y', which='major')


    print("done ",R0)
    R0 = 1.2
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[0, 1].plot(sigmas, error)
    axs[0, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[0, 1].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 1.5
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[1, 0].plot(sigmas, error)
    axs[1, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[1, 0].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 2
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[1, 1].plot(sigmas, error)
    axs[1, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[1, 1].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 4
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[2, 0].plot(sigmas, error)
    axs[2, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[2, 0].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 8
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[2, 1].plot(sigmas, error)
    axs[2, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[2, 1].grid(axis='y', which='major')
    print("done ", R0)
    plt.suptitle("Estimation error vs standard deviation (30 days rolling average) ", fontsize=14)

    plt.show()


def SingleNodeExamples():

    Gamma = 0.3
    fig, axs = plt.subplots(2, 2)


    for i in range(0,4):
        R0 = i+1
        Beta=Gamma*R0
        testNode = PopulationNode(1e06, name="Athens", beta=Beta, gamma=Gamma)
        testNode.TestInfect(10)
        testNode.advanceByDays(150)
        history = testNode.getHistory()
        x_axis = range(0,len(history))

        axs[int(i/2), i%2].plot(x_axis , [x[0] for x in history] , x_axis , [x[1] for x in history], x_axis, [x[2] for x in history])
        axs[int(i/2), i%2].set_title("R0=" + str(R0))
        axs[int(i/2), i%2].set_xlabel('Day')
        axs[int(i/2), i%2].set_ylabel('Population')
        axs[int(i/2), i%2].legend(['Susceptibles', 'Infectious', 'Removed'])


    plt.suptitle("Epidemic progress vs R0")
    plt.show()

