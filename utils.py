import math
import random
import pygame
import tkinter as tk
import numpy as np

def OHE(vector):    #np vector to list containing one 1 (argmax) and other 0s.
    indMax = np.argmax(vector)
    return [int(k == indMax) for k in range(len(vector))]