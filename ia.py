import tensorflow as tf
import math
import numpy as np
import random

import pygame
import tkinter as tk
from tkinter import messagebox
from utils import *

n_actions = 4
n_states = 12

ia_name = "policyGradients - RewardMean"



if ia_name == "randomPolicy":
	
	class Policy():

		def __init__(self):
			pass

		def __call__(self, state):
			chosenActionNumber = randrange(0,4)
			return np.array([int(k==chosenActionNumber) for k in range(4)])


if ia_name	== "policyGradients":

	class Policy():

		def __init__(self, newNN = False):

			print("Création de l'agent...")
			
			#Hyper-parameters
			self.lr = 0.1			#Learning rate
			self.saveN = 200		#Model saved every saveN steps.
			self.N = 20				#Batchsize, agent updates its gradients every N steps.
			self.optimizer = tf.keras.optimizers.Adam()

			#Initialize memory
			self.i = 0				#Step counter
			self.j = 0				#Step counter for saving model
			self.gradJ_sum = None			#Sum of all gradients on a batch
			self.gradLogPolicy_sum = None	#Sum of all policy's logarithm gradients on a game
			self.reward_sum = 0				#Sum of all rewards on a game

			#Our policy is a neural network
			if newNN:
				model = createModel()
			else: 
				try:
					model = tf.keras.models.load_model("model")
				except: 
					model = createModel()
			self.NN = model



		def __call__(self, state):
			with tf.GradientTape() as tape:

				#Compute action vector thanks to the NN
				actionVector = self.NN(np.array([state]))[0]		#[0.12   0   0   0.88]
				
				#Choose the action given a probability distribution pi(a|s)
				choosenAction = random.choices(range(n_actions), weights = actionVector)[0]
				actionVectorOHE = [int(k == choosenAction) for k in range(n_actions)]
				
				#Compute and save gradient of the logarithm of this probability.
				Pol = actionVector[choosenAction]
				logPol = tf.math.log(Pol)
				gradientsLogPol = tape.gradient(logPol, self.NN.trainable_variables)
				
				self.gradLogPolicy_sum = sumGradients(self.gradLogPolicy_sum, gradientsLogPol)


				#Return the action vector probabilistic (but one hot encoded...)
				return actionVectorOHE


		

		def getReward(self, reward):
			self.reward_sum += reward
			print("Reward obtenu pour aboutir à", self.reward_sum)


		def episodeEnd(self):
			self.i += 1
			self.j += 1

			#Compute gradJ for this episode
			gradJ = multiplyGradients(self.gradLogPolicy_sum, self.reward_sum)
			self.gradJ_sum = sumGradients(self.gradJ_sum, gradJ)

			#Improve policy if we have made N episodes
			if self.i == self.N:
				print("N épisodes éffectués, application des gradients...")
				minusGradJ = multiplyGradients(self.gradJ_sum, -1/self.N)
				self.optimizer.apply_gradients(zip(minusGradJ, self.NN.trainable_variables))

				#Reset memory 
				self.i = 0
				self.gradJ_sum = None

			#Reset memory for this episode, since gradient J has been computed, we now start a new episode and a new gradient
			self.reward_sum = 0
			self.gradLogPolicy_sum = None

			if self.j == self.saveN:
				print("SAVING...")
				self.NN.save("model")
				self.j = 0




		#UTILS METHODS
		def addElementToSum(self, sum, element):
			if sum == None:
				sum = element
			else:
				sum = sum + element
			return sum


if ia_name	== "policyGradients - Causal":

	class Policy():

		def __init__(self, newNN = False):

			print("Création de l'agent...")
			
			#Hyper-parameters
			self.lr = 0.1			#Learning rate
			self.saveN = 200		#Model saved every saveN steps.
			self.N = 20				#Batchsize, agent updates its gradients every N steps.
			self.optimizer = tf.keras.optimizers.Adam()

			#Initialize memory
			self.t = 0				#Instant
			self.i = 0				#Step counter (episode counter)
			self.j = 0				#Step counter for saving model

			self.listGradients_logPol = []
			self.listRewards = []		
			self.gradJ_sum = None

			#Our policy is a neural network
			if newNN:
				model = createModel()
			else: 
				try:
					model = tf.keras.models.load_model("model")
				except: 
					model = createModel()
			self.NN = model



		def __call__(self, state):
			with tf.GradientTape() as tape:

				self.t += 1

				#Compute action vector thanks to the NN
				actionVector = self.NN(np.array([state]))[0]		#[0.12   0   0   0.88]
				
				#Choose the action given a probability distribution pi(a|s)
				choosenAction = random.choices(range(n_actions), weights = actionVector)[0]
				actionVectorOHE = [int(k == choosenAction) for k in range(n_actions)]
				
				#Compute and save gradient of the logarithm of this probability.
				Pol = actionVector[choosenAction]
				logPol = tf.math.log(Pol)
				gradientsLogPol = tape.gradient(logPol, self.NN.trainable_variables)
				
				self.listGradients_logPol.append(gradientsLogPol)


				#Return the action vector probabilistic (but one hot encoded...)
				return actionVectorOHE


		

		def getReward(self, reward):
			self.listRewards.append(reward)
			print("Reward obtenu pour aboutir à", sum(self.listRewards))


		def episodeEnd(self):
			self.i += 1
			self.j += 1

			#Compute gradJ for this episode
			gradJ = None
			for tt in range(self.t):
				causalTotalReward = sum(self.listRewards[tt:])		#The causal reward is the sum of all rewards who were given after instant tt
				gradJ_toAdd = multiplyGradients(self.listGradients_logPol[tt], causalTotalReward)
				gradJ = sumGradients(gradJ, gradJ_toAdd)
			
			self.gradJ_sum = sumGradients(self.gradJ_sum, gradJ)

			#Improve policy if we have made N episodes
			if self.i == self.N:
				print("N épisodes éffectués, application des gradients...")

				minusGradJ = multiplyGradients(self.gradJ_sum, -1/self.N)
				self.optimizer.apply_gradients(zip(minusGradJ, self.NN.trainable_variables))

				#Reset memory 
				self.i = 0
				self.gradJ_sum = None

			#Save model if we have made saveN episodes
			if self.j == self.saveN:
				print("SAVING MODEL...")
				self.NN.save("model")
				self.j = 0

			#Reset memory for this episode, since gradient J has been computed, we now start a new episode and a new gradient
			self.listRewards = []
			self.listGradients_logPol = []
			self.t = 0


if ia_name	== "policyGradients - RewardMean":

	class Policy():

		def __init__(self, newNN = False):

			print("Création de l'agent...")
			
			#Hyper-parameters
			self.lr = 0.1			#Learning rate
			self.saveN = 200		#Model saved every saveN steps.
			self.N = 20				#Batchsize, agent updates its gradients every N steps.
			self.optimizer = tf.keras.optimizers.Adam()

			#Initialize memory
			self.t = 0				#Instant
			self.i = 0				#Step counter (episode counter)
			self.j = 0				#Step counter for saving model

			self.listListGradients_logPol = [[] for _ in range(self.N)]
			self.listListRewards = [[] for _ in range(self.N)]		

			#Our policy is a neural network
			if newNN:
				model = createModel()
			else: 
				try:
					model = tf.keras.models.load_model("model")
				except: 
					model = createModel()
			self.NN = model



		def __call__(self, state):
			with tf.GradientTape() as tape:

				self.t += 1

				#Compute action vector thanks to the NN
				actionVector = self.NN(np.array([state]))[0]		#[0.12   0   0   0.88]
				
				#Choose the action given a probability distribution pi(a|s)
				choosenAction = random.choices(range(n_actions), weights = actionVector)[0]
				actionVectorOHE = [int(k == choosenAction) for k in range(n_actions)]
				
				#Compute and save gradient of the logarithm of this probability.
				Pol = actionVector[choosenAction]
				logPol = tf.math.log(Pol)
				gradientsLogPol = tape.gradient(logPol, self.NN.trainable_variables)
				
				self.listListGradients_logPol[self.i].append(gradientsLogPol)


				#Return the action vector probabilistic (but one hot encoded...)
				return actionVectorOHE


		

		def getReward(self, reward):
			self.listListRewards[self.i].append(reward)
			print("Reward obtenu pour aboutir à", sum(self.listListRewards[self.i]))


		def episodeEnd(self):
			self.i += 1
			self.j += 1

		

			#Improve policy if we have made N episodes
			if self.i == self.N:
				print("N épisodes éffectués, application des gradients...")

				gradJ_sum = None
				for i in range(self.N):			#Pour chaque batch...
					gradJ_oneEpisode = None
					T = len(self.listListRewards[i])

					for t in range(T):	#Pour chaque instant t...
						meanReward = sum([sum(listReward) for listReward in self.listListRewards])/self.N
						centredTotalReward = sum(self.listListRewards[i]) - meanReward
						toAdd = multiplyGradients(self.listListGradients_logPol[i][t], centredTotalReward)
						gradJ_oneEpisode = sumGradients(gradJ_oneEpisode, toAdd)

					gradJ_sum = sumGradients(gradJ_sum, gradJ_oneEpisode)

				minusGradJ = multiplyGradients(gradJ_sum, -1/self.N)
				self.optimizer.apply_gradients(zip(minusGradJ, self.NN.trainable_variables))

				#Reset memory 
				self.i = 0
				self.listListRewards = [[] for _ in range(self.N)]
				self.listListGradients_logPol = [[] for _ in range(self.N)]

			#Save model if we have made saveN episodes
			if self.j == self.saveN:
				print("SAVING MODEL...")
				self.NN.save("model")
				self.j = 0

			#Reset memory for this episode, since gradient J has been computed, we now start a new episode and a new gradient
			self.t = 0
