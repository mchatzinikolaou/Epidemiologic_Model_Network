#import scipy.integrate as sc
#import numpy as np
from numpy import random
import numpy as np
# DONE
# Basic SIR in single Population
# Migration between populations
# Calculate traveling population categories (multinomial distribution in "travel packets")

# TODO
#Check design patterns

#Generalize the model : Add more parameters / arguments

# Regression
# Train a neural net

class PopulationNode:
    """
    Population Node.
    This class represents the progress in an isolated population has all the functionality to model the progress within
    the population , as well as the traffic of this population to and from others.

    """

    def __init__(self, total_Population,name, beta=0.3, gamma=0.05):
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

        #Demographics
        self.Population = total_Population
        self.S = 1.0
        self.I = 0.0
        self.R = 0.0

        self.b = beta
        self.g = gamma
        self.name = name
        self.history=[]

    # Returns populations as absolute values.
    def getTruePopulations(self):
        return [self.S * self.Population, self.I * self.Population, self.R * self.Population]
        # normalize to compensate for error


    # !!!!! REVISIT !!!! #
    def normalize(self, IntList):


        """
        Compensate for error.

        :param IntList:
        :return:
        """
        normalizedSIR=np.array(IntList)
        normalizedSIR=normalizedSIR/normalizedSIR.sum()
        return normalizedSIR

    #!!! REVISIT !!!

    # Simple SIR model so far.
    # This should be passed as a function in the constructor
    def model_SIR(self, SIR, t, beta, gamma):
        # Parameters
        S = SIR[0]
        I = SIR[1]
        R = SIR[2]

        # Model Equations
        dS = -beta * I * S
        dI = +beta * I * S - gamma * I
        dR = +gamma * I

        # Return final result.
        dSIR = [dS, dI, dR]
        return dSIR


    def advanceByDays(self, days=1):
        """
        Progress the epidemic by specified amount of days (defaults to 1 day per step).

        Changes S ,I , R and the rest of the parameters according to the specified model.
        (This should be , again , generalized)

        :param days: the amount of days we want to progress the solution.
        :return: the progress throughout these days.
        """
        z=[]
        S_old=self.S
        I_old=self.I
        R_old=self.R
        for day in range(0,days):

            #progress according to the model
            self.S+= -self.b*S_old*I_old
            self.I+= self.b*S_old*I_old-self.g*I_old
            self.R+= self.g*I_old

            #update history.
            z.append([round(self.S*self.Population),round(self.I*self.Population),round(self.R*self.Population)])

            #renew percentages for the next day.
            S_old = self.S
            I_old = self.I
            R_old = self.R

        self.history=self.history+z
        [self.S, self.I, self.R] = self.normalize([self.S, self.I, self.R])
        return z

    def getHistory(self):
        return self.history


    def assertCorrectPopulations(self):
        print(self.I+self.S+self.R)
        assert self.I+self.S+self.R==1, "Populations don't sum up to 1"



    def TravelTo(self, targetNode, groupSize):
        """
        :param targetNode: The node that will receive the travelers.
        :param groupSize: the total size of the travelling population
        """
        if(groupSize>=self.Population):
            print("Not enough passengers!")
            return

        [self.S, self.I, self.R] = self.normalize([self.S, self.I, self.R])

        #Create passenger batch
        travelers=random.multinomial(n=groupSize, pvals=[self.S,self.I,self.R])

        # infect the other node.
        dS = travelers[0]
        dI = travelers[1]
        dR = travelers[2]
        #print("Sending ",dS," susceptible, ",dI," infected and", dR," immune")
        targetNode.TravelFrom(dS,dI,dR)

        #remove passenger population from this node.
            #calculate using absolute values
        S_total = self.S * self.Population - dS
        if(S_total<0):
            S_total=0
        I_total = self.I * self.Population - dI
        if(I_total<0):
            I_total=0
        R_total = self.R * self.Population - dR
        if (R_total < 0):
            R_total = 0
        self.Population -= dS + dI + dR
            #update percentages.
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])




    def TravelFrom(self, dS, dI, dR):
        """
        Receive the population , seperated into groups of Infected, Susceptible and Resolved(immune)
        people.
        :param dS: Susceptible incoming people.
        :param dI: Infected
        :param dR: Immune
        """
        #Add absolute values
        S_total = self.S * self.Population +dS
        I_total =self.I * self.Population +dI
        R_total = self.R * self.Population +dR
        self.Population += dS+dI+dR

        #recalculate percentages
        [self.S , self.I, self.R] = self.normalize([S_total,I_total,R_total])





    # infect by a miniscule amount.
    def TestInfect(self, dI=100):
        """
        "Inject" the population with dI infected individuals. This is used to initiate
        the epidemic.

        :param dI:
        """
        S_total = int(self.S * self.Population) + 0
        I_total = int(self.I * self.Population) + dI
        R_total = int(self.R * self.Population) + 0
        self.Population += dI

        # recalculate percentages
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])


    def getName(self):
        return self.name


def Demonstrate1():
    """
    This is a demonstration of how Athens will fare with  those fake initial parameters of b and gamma.
    """
    testNode = PopulationNode(3875000, name="Athens")
    testNode.TestInfect()
    testNode.advanceByDays(275)
    print("Final demographics: \n")
    [S,I,R]=testNode.getTruePopulations()
    print("Susceptible: ", S)
    print("Infectious: ", I)
    print("Recovered: ", R)


def Demonstrate2():
    """
    A demonstration where the epidemic progresses for  70 days in Athens and then a travelling pack
    of 100 people travel to Chania.
    """
    testNode1 = PopulationNode(3875000, name="Athens")
    testNode2 = PopulationNode(50000, name="Chania")
    testNode1.TestInfect()
    testNode1.advanceByDays(70)
    testNode1.TravelTo(testNode2,100)
    testNode2.advanceByDays(100)

