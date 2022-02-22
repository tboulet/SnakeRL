import math
import numpy as np
import random

import pygame
import tkinter as tk
from tkinter import messagebox

n_actions = 4

def OHE(vector):    #np vector to list containing one 1 (argmax) and other 0s.
    indMax = np.argmax(vector)
    return [int(k == indMax) for k in range(len(vector))]

def sumListOfGradients(listGradients):
	listGradientsPure = [gradient for gradient in listGradients if gradient != None]
	if listGradientsPure == []: return None
	return [sum([gradient[k] for gradient in listGradientsPure]) for k in range(len(listGradientsPure[0]))]

def multiplyGradients(grad, factor):
	if grad == None:
		return None
	return [elem * factor for elem in grad]

def actionToVector(action):
	actionVector = [0 for _ in range(n_actions)]
	actionVector[action] = 1	
	return np.array(actionVector)

def actionToNumber(vector):
	return np.argmax(vector)

def callModel(model, input):
	#For model being V and input being s, return the float V(s) 
	return model(np.array([input]))[0] 