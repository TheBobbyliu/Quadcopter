import tensorflow as tf
import numpy as np
import sys
import pandas as pd
from task import Task
from agents.agent import DDPG
from collections import deque
import win_unicode_console

if __name__ == '__main__':
    # quadcopter parameters
    win_unicode_console.enable()                                  
    init_pose = np.array([0., 0., 0., 0., 0., 0.])  
    init_velocities = np.array([0., 0., 0.])        
    init_angle_velocities = np.array([0., 0., 0.])
    # 起飞
    target_pos = np.array([0., 0., 5.])
    runtime = 50
    # Setup
    task = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
    agent = DDPG(task)
    # hyperparameters
    agent.TAU = 0.2
    agent.a_lr = 0.01
    agent.c_lr = 0.04
    agent.buffer_size = 10000
    agent.hidden_size = 64 # hidden layer size
    agent.batch_size = 64
    agent.theta = 0.15
    agent.sigma = 0.3
    agent.gamma = 0.9
    agent.runtime = runtime
    agent.num_episodes = 1000
    agent.train()