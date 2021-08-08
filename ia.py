from random import *
import numpy as np
import tensorflow as tf
import math 

n_actions = 4
n_states = 1000

ia_name = "policyGradients"



if ia_name == "basicAlgo":
	
	class Policy():

		def __init__(self):
			pass

		def __call__(self, state):
			chosenActionNumber = randrange(0,4)
			return np.array([int(k==chosenActionNumber) for k in range(4)])

	class Memory():

		def __init__(self):
			pass

if ia_name	== "policyGradients":

	class Policy():

		def __init__(self):
			#Hyper-parameters
			self.lr = 0.1
			self.N = 3
			self.causality = False
			self.optimizer = tf.keras.optimizers.Adam()

			#Initialize memory
			self.i = 0
			self.gradJ_sum = None			#Mean of all gradients
			self.gradLogPolicy_sum = None	#Sum of logarithm of policy
			self.reward_sum = None			#Sum of rewards

			#Our policy is a neural network
			model = tf.keras.models.Sequential()
			model.add(tf.keras.layers.Dense(5, activation='relu'))
			model.add(tf.keras.layers.Dense(n_actions, activation='softmax'))
			self.NN = model



		def __call__(self, state):
			with tf.GradientTape() as tape:

				#Compute action vector thanks to the NN
				actionVector = self.NN(np.array([state]))[0]		#[0.12   0   0   0.88]
				
				#Choose the action given a probability distribution pi(a|s)
				choosenAction = choices(range(n_actions), weights = actionVector)[0]
				actionVectorOHE = [int(k == choosenAction) for k in range(n_actions)]
				
				#Compute and save gradient of the logarithm of this probability.
				Pol = actionVector[choosenAction]
				logPol = tf.math.log(Pol)
				gradientsLogPol = tape.gradient(logPol, self.NN.trainable_variables)
				self.gradLogPolicy_sum = self.addElementToSum(self.gradLogPolicy_sum, gradientsLogPol)

				#Return the action vector probabilistic (but one hot encoded...)
				return actionVectorOHE


		

		def getReward(self, reward):
			self.reward_sum = self.addElementToSum(self.reward_sum, reward)


		def episodeEnd(self):
			self.i += 1

			#Compute gradJ for this episode
			gradJ = self.reward_sum * self.gradLogPolicy_sum #####################################
			self.gradJ_sum = self.addElementToSum(self.gradJ_sum, gradJ)

			#Improve policy if we have made N episodes
			if self.i == self.N:
				gradJ = self.gradJ_sum/self.N
				self.optimizer.apply_gradients(zip(-gradJ, self.NN.trainable_variables))

				#Reset memory 
				self.i = 0
				self.gradJ_sum = None
		
			#Reset memory for this episode, since gradient J has been computed, we know start a new episode and a new gradient
			self.reward_sum = None
			self.gradLogPolicy_sum = None




		#UTILS METHODS
		def addElementToSum(self, sum, element):
			if sum == None:
				sum = element
			else:
				sum = sum + element
			return sum