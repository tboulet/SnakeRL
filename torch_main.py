#Torch for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary
#Python library
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
#Gym for environments, WandB for feedback
import gym
import wandb
from snake_env import SnakeEnv, rows
#RL agents
from div.utils import *
from rl_algos._ALL_AGENTS import REINFORCE, DQN, ACTOR_CRITIC, PPO
from rl_algos.AGENT import RANDOM_AGENT


def run(agent, env, steps, wandb_cb = True, 
        n_render = 20
        ):
    '''Train an agent on an env.
    agent : an AGENT instance (with methods act, learn and remember implemented)
    env : a gym env (with methods reset, step, render)
    steps : int, number of steps of training
    wandb_cb : bool, whether metrics are logged in WandB
    n_render : int, one episode on n_render is rendered
    '''
    
    print("Run starts.")
################### FEEDBACK #####################
    if wandb_cb: 
        try:
            from config import project, entity
        except ImportError:
            raise Exception("For ou need to specify your WandB ids in config.py\nConfig template is available at div/config_template.py")
        run = wandb.init(project=project, 
                        entity=entity,
                        config=agent.config,
        )
##################### END FEEDBACK ###################
    episode = 1
    step = 0
    while step < steps:
        done = False
        obs = env.reset()
        
        
        while not done and step < steps:
            action = agent.act(obs)                                                 #Agent acts
            next_obs, reward, done, info = env.step(action)                         #Env reacts          
            agent.remember(obs, action, reward, done, next_obs, info)    #Agent saves previous transition in its memory
            agent.learn()                                                #Agent learn (eventually)
            
            ###### Feedback ######
            print(f"Episode n°{episode} - Total step n°{step} ...", end = '\r')
            if episode % n_render == 0:
                env.render()
            if wandb_cb:
                agent.log_metrics()
            ######  End Feedback ######  

            #If episode ended, reset env, else change state
            if done:
                step += 1
                episode += 1
                break
            else:
                step += 1
                obs = next_obs
    
    if wandb_cb: run.finish()   #End wandb run.
    print("End of run.")
    
    
    

if __name__ == "__main__":
    #ENV
    env = SnakeEnv()
    n_actions = 4
    n_flatten = (rows - 2 - 2)**2
    
    #ACTOR PI
    actor =  nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),     
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_flatten * 64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(),
        )
    
    #CRITIC Q
    action_value =  nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),     
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_flatten * 64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    #STATE VALUE V
    state_value = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),     
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_flatten * 64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    #AGENT
    dqn = DQN(action_value=action_value)
    reinforce = REINFORCE(actor=actor)
    ppo = PPO(actor = actor, state_value = state_value)
    ac = ACTOR_CRITIC(actor = actor, state_value = state_value)
    random_agent = RANDOM_AGENT(2)
    
    agent = dqn
    
    #RUN
    run(agent, 
        env = env, 
        steps=500000, 
        wandb_cb = False,
        n_render=20,
        )    