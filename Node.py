import scipy.integrate as sc
import numpy as np
import matplotlib.pyplot as plt



#DONE
# Basic SIR in single Population
# Migration between populations (missing travel categories of populations)


# TODO
# Calculate traveling population categories (multinomial distribution in "travel packets")
# Regression
# Train a neural net

#A Single epidemiologic node. This will hold the advancement of the epidemic.
#As of now it implemets
# . Simple SIR model
# . Migration between nodes

class PopulationNode:

    def __init__(self, Total_Population, beta=0.3, gamma=0.1, name="N/A",Epidemiologic_Model="asd", NeighbourMap="asd"):
        # Normalize values so that the compartmental model is solid.
        self.Population = Total_Population
        self.S = 1.0
        self.I = 0.0
        self.R = 0.0
        self.b = beta
        self.g = gamma
        self.name=name

    def getTruePopulations(self):
        return [self.S*self.Population, self.I*self.Population,self.R*self.Population]

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
    #Returns the population(day) matrices
    def advanceByDays(self, days):
        # Time variable
        t = np.linspace(0, days)

        # solve the model
        z = sc.odeint(self.model_SIR, [self.S, self.I, self.R], t, args=(self.b, self.g,))

        # update populations
        self.S = z[-1, 0]
        self.I = z[-1, 1]
        self.R = z[-1, 2]

        # show results, for debugging etc.
        plt.figure(str(self))
        plt.plot(t, z[:, 0], 'r-')
        plt.plot(t, z[:, 1], 'g+')
        plt.plot(t, z[:, 2], 'b:')
        plt.show()
        return z


    # A batch of <batchSize> people travels to target node.
    # This will follow to binomial probability, for the <infected_individuals> out of <batch_size>
    # This could be modelled using Monte Carlo simulations, and is yet an open question.


    #INCOMPLETE
    def emmigrateTo(self, targetNode, batchSize):

        #de-normalize
        #This may need some integerization
        self.S = self.S * self.Population
        self.I = self.I * self.Population
        self.R = self.R * self.Population
        # Calculate people that are on board . (this varies according to the probabilistic model - probably a Categorical/Multinomial Distribution).

        # I_Probability =   #Calculates the probability of any individual aboard being infected.
        I_travellers = 5

        # R_Probability =   #Calculates the probability of any individual aboard being infected.
        R_travellers = 0

        # S_Probability =   #Calculates the probability of any individual aboard being infected.
        S_travellers = 0

        # infect the other node.
        targetNode.immigrateFrom(S_travellers,I_travellers,R_travellers)

        # update populations
        # This may need some integerization
        self.R -= R_travellers
        self.I -= I_travellers
        self.S -= S_travellers
        self.Population -= batchSize


        #normalize
        self.S = self.S / self.Population
        self.I = self.I / self.Population
        self.R = self.R / self.Population

        # Add dI infected people in the population

    #INCOMPLETE
    def immigrateFrom(self, dS,dI,dR):
        # de-normalize
        # This may need some integerization
        self.S = self.S * self.Population
        self.I = self.I * self.Population
        self.R = self.R * self.Population

        # This may need some integerization
        self.R += dR
        self.I += dI
        self.S += dS
        self.Population += dR+dI+dS

        # normalize
        self.S = self.S / self.Population
        self.I = self.I / self.Population
        self.R = self.R / self.Population

    def TestInfect(self,dI=2):
        self.Population += dI
        self.I+=dI/self.Population

"""
# Demonstrating :
node1 = PopulationNode(11000000, 0.5, 0.1)
node2 = PopulationNode(1000, 0.2, 0.1)

node1.TestInfect()
node1.advanceByDays(100)
node1.emmigrateTo(node2,10)
node2.advanceByDays(100)
"""
