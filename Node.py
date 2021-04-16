# import scipy.integrate as sc

from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
"""def RunUniformNoiseTests():
    R0 = 1

    fig, axs = plt.subplots(3, 2)

    [error, range] = UniformNoiseTest(truebeta=np.multiply(R0, 0.05), K_max=1)
    axs[0, 0].semilogy(range, error)
    axs[0, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[0, 0].grid(axis='y', which='major')

    R0 = 1.2
    [error, range] = UniformNoiseTest(truebeta=np.multiply(R0, 0.05), K_max=1)
    axs[0, 1].semilogy(range, error)
    axs[0, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[0, 1].grid(axis='y', which='major')

    R0 = 1.5
    [error, range] = UniformNoiseTest(truebeta=np.multiply(R0, 0.05), K_max=1)
    axs[1, 0].semilogy(range, error)
    axs[1, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[1, 0].grid(axis='y', which='major')

    R0 = 2
    [error, range] = UniformNoiseTest(truebeta=np.multiply(R0, 0.05), K_max=1)
    axs[1, 1].semilogy(range, error)
    axs[1, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[1, 1].grid(axis='y', which='major')

    R0 = 4
    [error, range] =UniformNoiseTest(truebeta=np.multiply(R0, 0.05), K_max=1)
    axs[2, 0].semilogy(range, error)
    axs[2, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[2, 0].grid(axis='y', which='major')

    R0 = 8
    [error, range] =UniformNoiseTest(truebeta=np.multiply(R0, 0.05), K_max=1)
    axs[2, 1].semilogy(range, error)
    axs[2, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[2, 1].grid(axis='y', which='major')

    plt.suptitle("Estimation error vs noise amplitude", fontsize=14)

    plt.show()
"""



"""def UniformNoiseTest(truebeta=0.2,K_max=0.4):
    testNode = PopulationNode(3875000, name="Athens", beta=truebeta)

    [S_b, I_b, R_b] = [15, 3, 1]
    testNode.biasedAdvanceByDays(S_b, I_b, R_b, 1000)
    print(testNode.b)
    history = testNode.getHistory()
    testNode.plotNodeHistory()
    # calculate true beta
    print("estimated beta: ", calculateBetaBiased(history, S_b))
    # calculate betas with noise
    betas = []
    ranges = np.linspace(0, K_max, 400)
    for repeat in range(0, 20):
        iter_betas = []
        for K in ranges:
            noiseHistory = addNoise(history, S_b, I_b, R_b, type='uniform',k=K)

            S = noiseHistory[0][:]
            I = noiseHistory[1][:]
            R = noiseHistory[2][:]
            noiseHistory = [[S[i], I[i], R[i]] for i in range(0, len(S))]
            iter_betas.append(calculateBetaBiased(noiseHistory, S_b))

        betas.append(iter_betas)
    betas = np.mean(betas, axis=0)

    error = np.divide(np.subtract(betas, truebeta), truebeta)
    return [np.abs(error),ranges]
"""


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

            self.TravelFrom(S_bias, I_bias, R_bias)
            # update history.
            z.append([self.S * self.Population, self.I * self.Population, self.R * self.Population])

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
            print("Not enough passengers!")
            return

        [self.S, self.I, self.R] = self.normalize([self.S, self.I, self.R])

        # Create passenger batch
        travelers = random.multinomial(n=groupSize, pvals=[self.S, self.I, self.R])

        # infect the other node.
        dS = travelers[0]
        dI = travelers[1]
        dR = travelers[2]
        # print("Sending ",dS," susceptible, ",dI," infected and", dR," immune")
        infected = targetNode.TravelFrom(dS, dI, dR)

        # remove passenger population from this node.
        # calculate using absolute values
        S_total = self.S * self.Population - dS
        if S_total < 0:
            S_total = 0
        I_total = self.I * self.Population - dI
        if I_total < 0:
            I_total = 0
        R_total = self.R * self.Population - dR
        if R_total < 0:
            R_total = 0
        self.Population -= dS + dI + dR
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
        S_total = self.S * self.Population + dS

        I_total = self.I * self.Population + dI

        R_total = self.R * self.Population + dR
        self.Population += dS + dI + dR

        # recalculate percentages
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])
        return infected

    def TestInfect(self, dI=100):
        """
        "Inject" the population with dI infected individuals. This is used to initiate
        the epidemic.

        :param dI:
        """
        S_total = int(self.S * self.Population + 0.5) + 0
        I_total = int(self.I * self.Population + 0.5) + dI
        R_total = int(self.R * self.Population + 0.5) + 0
        self.Population += dI

        # recalculate percentages
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])

    def getName(self):
        return self.name

    def getHistory(self):
        return self.history

    def getTruePopulations(self):
        return [int(self.S * self.Population + 0.5), int(self.I * self.Population + 0.5),
                int(self.R * self.Population + 0.5)]

    def updatePopulation(self):
        self.PopulationHistory.append(self.Population)

    def normalize(self, IntList):

        normalizedSIR = np.array(IntList)
        size = len(normalizedSIR)
        normalizedSIR = normalizedSIR / normalizedSIR.sum()
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
            break
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
        beta_calculated.append(
            np.multiply(np.divide(bS, denom * population) - np.divide(St_init - St_1, denom), population))
    return np.median(beta_calculated)

def addNoise(history, bS, bI, bR, kS=1.0, kI=1.0, kR=1.0,type='normal',k=1.0):

    S = [i[0] for i in history]
    I = [i[1] for i in history]
    R = [i[2] for i in history]

    # Add noise to the data
    Snoise=0
    Inoise=0
    Rnoise=0
    if type=='normal':
        Snoise = np.random.normal(loc=0, scale=kS * bS, size=len(S))
        Inoise = np.random.normal(loc=0, scale=kI * bI, size=len(I))
        Rnoise = np.random.normal(loc=0, scale=kR * bR, size=len(R))
    elif type == "uniform":
        Snoise = np.multiply(np.random.uniform(low=-k,high=k, size=len(S)),S)
        Inoise = np.multiply(np.random.uniform(low=-k,high=k, size=len(I)),I)
        Rnoise = np.multiply(np.random.uniform(low=-k,high=k, size=len(R)),R)

    S = np.add(S, Snoise)
    I = np.add(I, Inoise)
    R = np.add(R, Rnoise)

    # Return the data with noise
    return [S, I, R]


def addUniformNoise(history,sigma=1):
    S=[i[0] for i in history]
    I=[i[1] for i in history]
    R=[i[2] for i in history]
    [Sn,In,Rn]=[[],[],[]]
    for i in range(0,len(S)):
        Sn.append( np.add(S[i],np.random.normal(loc=0, scale=sigma*S[i])))
        In.append( np.add(I[i],np.random.normal(loc=0, scale=sigma*I[i])))
        Rn.append( np.add(R[i],np.random.normal(loc=0, scale=sigma*R[i])))
    return [Sn,In,Rn]


def NormalNoiseTest(truebeta=0.2,max_sigma=2):
    testNode = PopulationNode(3875000, name="Athens", beta=truebeta)

    [S_b, I_b, R_b] = [15, 3, 1]
    testNode.biasedAdvanceByDays(S_b, I_b, R_b, 500)
    print("True beta: ", testNode.b)
    history = testNode.getHistory()

    # calculate true beta
    print("estimated beta: ", calculateBetaBiased(history, S_b))

    # calculate betas with noise
    betas = []
    sigmas = np.linspace(0,max_sigma,200)
    for repeat in range(0,50):
        iter_betas = []
        for k in sigmas:
            noiseHistory = addUniformNoise(history,sigma=k)

            S = noiseHistory[0][:]
            I = noiseHistory[1][:]
            R = noiseHistory[2][:]

            #take rolling averages

            M=30
            S = uniform_filter1d(S, size=M)
            I = uniform_filter1d(I, size=M)
            R = uniform_filter1d(R, size=M)

            noiseHistory = [[S[i], I[i], R[i]] for i in range(0, len(S))]
            iter_betas.append(calculateBetaBiased(noiseHistory, S_b))

        betas.append(iter_betas)
    betas = np.mean(betas, axis=0)
    error =np.abs( np.divide(np.subtract(betas, truebeta), truebeta))
    return [error, sigmas]





def RunNormalNoiseTests():

    R0 = 1
    [error, sigmas] = NormalNoiseTest(truebeta=np.multiply(R0, 0.05), max_sigma=0.1)
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(sigmas, error)
    axs[0, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[0, 0].grid(axis='y', which='major')


    print("done ",R0)
    R0 = 1.2
    [error, sigmas] = NormalNoiseTest(truebeta=np.multiply(R0, 0.05), max_sigma=0.5)
    axs[0, 1].plot(sigmas, error)
    axs[0, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[0, 1].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 1.5
    [error, sigmas] = NormalNoiseTest(truebeta=np.multiply(R0, 0.05), max_sigma=3)
    axs[1, 0].plot(sigmas, error)
    axs[1, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[1, 0].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 2
    [error, sigmas] = NormalNoiseTest(truebeta=np.multiply(R0, 0.05), max_sigma=2)
    axs[1, 1].plot(sigmas, error)
    axs[1, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[1, 1].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 4
    [error, sigmas] = NormalNoiseTest(truebeta=np.multiply(R0, 0.05), max_sigma=4)
    axs[2, 0].plot(sigmas, error)
    axs[2, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[2, 0].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 8
    [error, sigmas] = NormalNoiseTest(truebeta=np.multiply(R0, 0.05), max_sigma=10)
    axs[2, 1].plot(sigmas, error)
    axs[2, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[2, 1].grid(axis='y', which='major')
    print("done ", R0)
    plt.suptitle("Estimation error vs standard deviation (30 days rolling average) ", fontsize=14)

    plt.show()

#RunNormalNoiseTests()

