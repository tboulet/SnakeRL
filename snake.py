#Snake Tutorial Python
import tensorflow as tf
import math
import numpy as np
import random

import pygame
import tkinter as tk
from tkinter import messagebox

from utils import *


PLAYER_TYPE = "IA"



if PLAYER_TYPE == "IA": from ia import Policy

class cube(object):    
    global rows, width

    def __init__(self,start,dirnx=1,dirny=0,color=(255,0,0)):
        self.pos = start
        self.rows = rows
        self.w = width
        self.dirnx = 1
        self.dirny = 0
        self.color = color
 
       
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)
 
    def draw(self, surface, eyes=False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]
 
        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)
       
 
class snake(object):
    body = []
    turns = {}
    def __init__(self, color, pos):
        self.color = color
        self.head = cube(pos)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
 
    def move(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            

        #Choose action
        if PLAYER_TYPE == "HUMAN":
            keys = pygame.key.get_pressed()

        elif PLAYER_TYPE == "IA":
            #We here let the AI read the current state, do what it has to do (learn?, memorize?) and ask it to answer with an action vector
            state = readState()
            actionsVectorOneHotEncoded = policy(state)  #[0 0 1 0]
            keyNumbers = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]
            keys = {keyNumbers[k] : actionsVectorOneHotEncoded[k] for k in range(4)}
            


        # Move accordingly to  action chosen
        if keys[pygame.K_LEFT]:
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif keys[pygame.K_RIGHT]:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif keys[pygame.K_UP]:
            self.dirnx = 0
            self.dirny = -1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif keys[pygame.K_DOWN]:
            self.dirnx = 0
            self.dirny = 1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        #Move the entire snake body
        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]

                c.dirnx, c.dirny = turn
                if c.dirnx == -1 and c.pos[0] <= 0: c.pos = (c.rows-1, c.pos[1])
                elif c.dirnx == 1 and c.pos[0] >= c.rows-1: c.pos = (0,c.pos[1])
                elif c.dirny == 1 and c.pos[1] >= c.rows-1: c.pos = (c.pos[0], 0)
                elif c.dirny == -1 and c.pos[1] <= 0: c.pos = (c.pos[0],c.rows-1)
                else: c.move(c.dirnx,c.dirny)

                if i == len(self.body)-1:
                    self.turns.pop(p)
            else:
                if c.dirnx == -1 and c.pos[0] <= 0: c.pos = (c.rows-1, c.pos[1])
                elif c.dirnx == 1 and c.pos[0] >= c.rows-1: c.pos = (0,c.pos[1])
                elif c.dirny == 1 and c.pos[1] >= c.rows-1: c.pos = (c.pos[0], 0)
                elif c.dirny == -1 and c.pos[1] <= 0: c.pos = (c.pos[0],c.rows-1)
                else: c.move(c.dirnx,c.dirny)
       

 
    def reset(self, pos):
        self.head = cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
 
 
    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny
 
        if dx == 1 and dy == 0:
            self.body.append(cube(((tail.pos[0]-1)%rows,(tail.pos[1])%rows)))
        elif dx == -1 and dy == 0:
            self.body.append(cube(((tail.pos[0]+1)%rows,(tail.pos[1])%rows)))
        elif dx == 0 and dy == 1:
            self.body.append(cube(((tail.pos[0])%rows,(tail.pos[1]-1)%rows)))
        elif dx == 0 and dy == -1:
            self.body.append(cube(((tail.pos[0])%rows,(tail.pos[1]+1)%rows)))
 
        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy
       
 
    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i ==0:
                c.draw(surface, True)
            else:
                c.draw(surface)
 
 
def drawGrid(w, rows, surface): 
    sizeBtwn = w // rows
 
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn
 
        pygame.draw.line(surface, (255,255,255), (x,0),(x,w))
        pygame.draw.line(surface, (255,255,255), (0,y),(w,y))
       
 
def redrawWindow(surface):
    global rows, width, s, snack
    surface.fill((0,0,0))
    s.draw(surface)
    snack.draw(surface)
    drawGrid(width,rows, surface)
    pygame.display.update()
 
 
def randomSnack(rows, item):
 
    positions = item.body
 
    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z:z.pos == (x,y), positions))) > 0:
            continue
        else:
            break
       
    return (x,y)
 
 
def message_box(subject, content):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    messagebox.showinfo(subject, content)
    try:
        root.destroy()
    except:
        pass
 

def readState():
    global s, snack, rows, width
    arrayHead = np.array(np.zeros((rows, rows)))
    arrayBody = np.array(np.zeros((rows, rows)))
    arraySnack = np.array(np.zeros((rows, rows)))
    
    arrayHead[s.head.pos[0],s.head.pos[1]] = 1
    for bodyPart in s.body:
        arrayBody[bodyPart.pos[0], bodyPart.pos[1]] = 1
    arraySnack[snack.pos[0], snack.pos[1]] = 1

    return np.vstack((arrayHead, arrayBody, arraySnack)).flatten()




delay = True
if PLAYER_TYPE == "IA": policy = Policy(newNN = True)

def main():
    global width, rows, s, snack

    width = 500
    rows = 4
    win = pygame.display.set_mode((width, width))
    s = snake((255,0,0), (rows//2,rows//2))
    snack = cube(randomSnack(rows, s), color=(0,255,0))
    flag = True
    gameEnd = False
    clock = pygame.time.Clock()
   
    while flag:
        if delay: pygame.time.delay(50)
        #clock.tick(10)

        #Here,  s.move() will call for the policy, given the current state.
        s.move()

        #If the snake meet a snack, he gains a piece of body, a new snack is generated and AI get rewarded of 1.
        if s.body[0].pos == snack.pos:
            if len(s.body) >= rows**2 - 2: #If map is full of the body, game end
                gameEnd = True
            s.addCube()
            snack = cube(randomSnack(rows, s), color=(0,255,0))
            if PLAYER_TYPE == "IA": policy.getReward(1)
        else:
            if PLAYER_TYPE == "IA": policy.getReward(0)
 
        #If snack meet his own body, the game end.
        for x in range(len(s.body)):
            if gameEnd or s.body[x].pos in list(map(lambda z:z.pos,s.body[x+1:])):
                print(f'Fin de la partie, score: , {len(s.body)}')
                if PLAYER_TYPE == "HUMAN": message_box('You Lost!', 'Play again...')
                if PLAYER_TYPE == "IA": policy.episodeEnd()
                s.reset((rows//2,rows//2))
                gameEnd = False
                break
 
           
        redrawWindow(win)
 
       
    pass
 
 
 
main()