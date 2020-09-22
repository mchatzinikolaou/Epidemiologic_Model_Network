#import scipy.integrate as sc
#import numpy as np
from numpy import random

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

    def __init__(self, total_Population,name, beta=0.3, gamma=0.1):
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
        for day in range(0,days):
            self.S+= -self.b*self.I*self.S
            self.I+= self.b*self.I*self.S-self.g*self.I
            self.R+= self.g*self.I
            z.append([int(self.S*self.Population),int(self.I*self.Population),int(self.R*self.Population)])

        self.history=self.history+z

        return z

    def getHistory(self):
        return self.history

    def TravelTo(self, targetNode, groupSize):
        """
        :param targetNode: The node that will receive the travelers.
        :param batchSize: the total size of the travelling population
        """
        if(groupSize>=self.Population):
            print("Not enough passengers!")
            return

        #Create passenger batch
        travelers=random.multinomial(n=groupSize, pvals=[self.S,self.I,self.R])

        # infect the other node.
        dS = travelers[0]
        dI = travelers[1]
        dR = travelers[2]
        print("Sending ",dS," susceptible, ",dI," infected and", dR," immune")
        targetNode.TravelFrom(dS,dI,dR)

        #remove passenger population from this node.
            #calculate using absolute values
        S_total = int(self.S * self.Population) - dS
        I_total = int(self.I * self.Population) - dI
        R_total = int(self.R * self.Population) - dR
        self.Population -= dS + dI + dR
            #update percentages.
        self.S = S_total / self.Population
        self.I = I_total / self.Population
        self.R = R_total / self.Population


    def TravelFrom(self, dS, dI, dR):
        """
        Receive the population , seperated into groups of Infected, Susceptible and Resolved(immune)
        people.
        :param dS: Susceptible incoming people.
        :param dI: Infected
        :param dR: Immune
        """
        #Add absolute values
        S_total = int(self.S * self.Population) +dS
        I_total = int(self.I * self.Population) +dI
        R_total = int(self.R * self.Population) +dR
        self.Population += dS+dI+dR

        #recalculate percentages
        self.S = S_total / self.Population
        self.I = I_total/self.Population
        self.R = R_total/self.Population
        # normalize to compensate for error



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
        self.S = S_total / self.Population
        self.I = I_total / self.Population
        self.R = R_total / self.Population

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

