from time import sleep
import gym
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
from utils import *

n_actions = 4
rows = 5
width = 500

snackReward = 30
nothingReward = -1
deathReward = 0
delay = 200
GREEN = (0,255,0)
RED = (255,0,0)



class cube(object):    

    def __init__(self,start,dirnx=1,dirny=0,color=(255,0,0)):
        self.pos = start
        self.w = width
        self.rows = rows
        self.dirnx = 1
        self.dirny = 0
        self.color = color
 
       
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)
 
    def draw(self, surface, eyes=False):
        dis = self.w // rows
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
 
    def move(self, action):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            
        #Choose action
        keyNumbers = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]
        keys = {keyNumbers[k] : int(k == action) for k in range(n_actions)}

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
 



class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super().__init__()
        self.win = pygame.display.set_mode((width, width))
        self.snake = snake(RED, (rows//2,rows//2))
        self.snack = cube(randomSnack(rows, self.snake), color=GREEN)
        
    
    def step(self, action):
        done = False
        s = self.snake

        s.move(action)
        #If snack meet his own body, the game end.
        for x in range(len(s.body)): 
            if s.body[x].pos in list(map(lambda z:z.pos,s.body[x+1:])):
                reward = deathReward
                s.reset((rows//2,rows//2))
                done = True
                break

        #If the snake meet a snack, he gains a piece of body, a new snack is generated and AI get rewarded.
        if not done and s.body[0].pos == self.snack.pos:
            if len(s.body) >= rows**2 - 2: #If map is full of the body, game end.
                done = True
            s.addCube()
            self.snack = cube(randomSnack(rows, s), color=GREEN)
            reward = snackReward
        else:
            reward = nothingReward

        next_obs = self.readState()
        info = dict()
        return next_obs, reward, done, info


    def reset(self):
        self.snake.reset((rows//2,rows//2))
        obs = self.readState()
        return obs
    
    def render(self):
        surface = self.win
        surface.fill((0,0,0))
        self.snake.draw(surface)
        self.snack.draw(surface)
        drawGrid(width,rows, surface)
        pygame.display.update()
        sleep(0.02)
    
    def readState(self):
        arr = np.array(np.zeros((3, rows, rows)), dtype=np.float32)
        s = self.snake        
        arr[0, s.head.pos[0],s.head.pos[1]] = 1.
        for bodyPart in s.body:
            arr[1, bodyPart.pos[0], bodyPart.pos[1]] = 1.
        arr[2, self.snack.pos[0], self.snack.pos[1]] = 1.
        return arr
    
        
    