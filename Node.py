import scipy.integrate as sc
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


# DONE
# Basic SIR in single Population
# Migration between populations
# Calculate traveling population categories (multinomial distribution in "travel packets")

# TODO
#Change names
#Check design patterns

#Generalize the model : Add more parameters / arguments

# Regression
# Train a neural net

class PopulationNode:

    #The node models some compartmental model.
    def __init__(self, total_Population,name, beta=0.3, gamma=0.1):
        # Normalize values so that the compartmental model is solid.
        self.Population = total_Population
        self.S = 1.0
        self.I = 0.0
        self.R = 0.0
        self.b = beta
        self.g = gamma
        self.name = name
        self.history=[]

    # Returns true populations
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

    # Advance the spread by <days>
    # Returns the matrices with the given populations at the end of the advancement.
    def advanceByDays(self, days):



        z=[]
        for day in range(0,days):
            self.S+= -self.b*self.I*self.S
            self.I+= self.b*self.I*self.S-self.g*self.I
            self.R+= self.g*self.I
            z.append([int(self.S*self.Population),int(self.I*self.Population),int(self.R*self.Population)])

        # Time variable
        # t = range(0, days)
        # solve the model
        # z = sc.odeint(self.model_SIR, [self.S, self.I, self.R], t, args=(self.b, self.g,))
        #[self.S,self.I,self.R]=z[-1]


        #self.history=self.history+z.tolist()
        self.history=self.history+z

        return z

    # A batch of <batchSize> people travels to target node.
    # This will follow to binomial probability, for the <infected_individuals> out of <batch_size>
    # This could be modelled using Monte Carlo simulations, and is yet an open question.

    def getHistory(self):
        return self.history

    def emmigrateTo(self, targetNode, batchSize):
        if(batchSize>=self.Population):
            print("Not enough passengers!")
            return

        #Create passenger batch
        travelers=random.multinomial(n=batchSize, pvals=[self.S,self.I,self.R])

        # infect the other node.
        dS = travelers[0]
        dI = travelers[1]
        dR = travelers[2]
        print("Sending ",dS," susceptible, ",dI," infected and", dR," immune")
        targetNode.immigrateFrom(dS,dI,dR)

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


    def immigrateFrom(self, dS, dI, dR):

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
    testNode = PopulationNode(3875000, name="Athens")
    testNode.TestInfect()
    testNode.advanceByDays(275)
    print("Final demographics: \n")
    [S,I,R]=testNode.getTruePopulations()
    print("Susceptible: ", S)
    print("Infectious: ", I)
    print("Recovered: ", R)


def Demonstrate2():
    testNode1 = PopulationNode(3875000, name="Athens")
    testNode2 = PopulationNode(50000, name="Chania")
    testNode1.TestInfect()
    testNode1.advanceByDays(70)
    testNode1.emmigrateTo(testNode2,100)
    testNode2.advanceByDays(100)

