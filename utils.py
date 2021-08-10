import tensorflow as tf
import math
import numpy as np
import random

import pygame
import tkinter as tk
from tkinter import messagebox


def OHE(vector):    #np vector to list containing one 1 (argmax) and other 0s.
    indMax = np.argmax(vector)
    return [int(k == indMax) for k in range(len(vector))]

def sumGradients(grad1, grad2):
	if grad1 == None:
		return grad2
	elif grad2 == None:
		return grad1
	else:
		return [grad1[k]+grad2[k] for k in range(len(grad1))]

def multiplyGradients(grad, factor):
	if grad == None:
		return None
	return [elem * factor for elem in grad]

def createModel():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(64, activation='relu'))		
	model.add(tf.keras.layers.Dense(32, activation='relu'))		
	model.add(tf.keras.layers.Dense(16, activation='relu'))	
	model.add(tf.keras.layers.Dense(4, activation='softmax'))
	return model