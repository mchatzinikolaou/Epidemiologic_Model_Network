import scipy.integrate as sc
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


# DONE
# Basic SIR in single Population
# Migration between populations
# Calculate traveling population categories (multinomial distribution in "travel packets")

# TODO



#Generalize the model : Add more parameters / arguments

# Regression
# Train a neural net

class PopulationNode:

    #The node models some compartmental model.
    def __init__(self, total_Population, beta=0.3, gamma=0.1, name="N/A"):
        # Normalize values so that the compartmental model is solid.
        self.Population = total_Population
        self.S = 1.0
        self.I = 0.0
        self.R = 0.0
        self.b = beta
        self.g = gamma
        self.name = name

    # Returns true populations
    def getTruePopulations(self):
        return [int(self.S * self.Population), int(self.I * self.Population),int( self.R * self.Population)]

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
        # Time variable
        t = np.linspace(0, days)

        # solve the model
        z = sc.odeint(self.model_SIR, [self.S, self.I, self.R], t, args=(self.b, self.g,))


        # show results, for debugging etc.
        plt.figure(str(self))
        plt.plot(t, z[:, 0], 'r-')
        plt.plot(t, z[:, 1], 'g-')
        plt.plot(t, z[:, 2], 'b-')
        plt.show()

        [self.S,self.I,self.R]=z[-1]
        return z

    # A batch of <batchSize> people travels to target node.
    # This will follow to binomial probability, for the <infected_individuals> out of <batch_size>
    # This could be modelled using Monte Carlo simulations, and is yet an open question.


    def emmigrateTo(self, targetNode, batchSize):

        #Calculate multinomial probabilities
        [s,i,r]=self.getTruePopulations()
        [ps,pi,pr]=[s/self.Population,i/self.Population,r/self.Population]

        #draw batch size individuals and add one to each category.
        passengers=random.multinomial(n=batchSize, pvals=[ps,pi,pr])

        # infect the other node.
        dS = passengers[0]
        dI = passengers[1]
        dR = passengers[2]
        targetNode.immigrateFrom(dS,dI,dR)

        # update populations
        # This may need some integerization

        self.S -= dS/self.Population
        self.I -= dI/self.Population
        self.R -= dR/self.Population
        self.Population -= batchSize


    # INCOMPLETE
    def immigrateFrom(self, dS, dI, dR):

        # This may need some integerization
        self.R += dR/self.Population
        self.I += dI/self.Population
        self.S += dS/self.Population
        self.Population += (dR + dI + dS)/self.Population


    # infect by a miniscule amount.
    def TestInfect(self, dI=2):
        self.Population += dI
        self.I += dI / self.Population

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

Demonstrate2()