import sys
sys.path.append('../')
from noise import OUNoise
from model import Actor, Critic
import buffer
import tensorflow as tf
import numpy as np
from collections import deque
import random
import time
from scipy.io import savemat

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = self.task.state_size
        self.action_size = self.task.action_size
        self.action_low = self.task.action_low
        self.action_high = self.task.action_high
        self.a_lr = 0.001
        self.c_lr = 0.002
        self.hidden_size = 32
        self.num_episodes = 10000
        self.runtime = 100
        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 32
        self.memory = buffer.Buffer(self.buffer_size, self.batch_size)
        # Ornstein-Uhlenbeck noise
        self.theta = 0.15
        self.sigma = 0.3
        self.noise = OUNoise(self.action_size, None, self.theta, self.sigma)
        # Algorithm parameters
        self.gamma = 0.9  # discount factor
        self.TAU = 0.01  # for soft update of target parameters

    def model_inputs(self):
        state = tf.placeholder(tf.float32,[None,self.task.state_size])
        action = tf.placeholder(tf.float32, [None, self.task.action_size])
        Qs = tf.placeholder(tf.float32, [None,1])
        actions_gradient = tf.placeholder(tf.float32, [None, self.task.action_size])
        return state, action, Qs, actions_gradient
    
    def model_opt(self, a_loss, c_loss):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            a_opt = tf.train.AdamOptimizer(self.a_lr).minimize(-a_loss)
            c_opt = tf.train.AdamOptimizer(self.c_lr).minimize(c_loss)
            return a_opt, c_opt

    def update_target(self, soft = True):
        """
        t_vars = tf.trainable_variables()
        al_vars = [var for var in t_vars if var.name.startswith('actor_local')]
        at_vars = [var for var in t_vars if var.name.startswith('actor_target')]
        cl_vars = [var for var in t_vars if var.name.startswith('critic_local')]
        ct_vars = [var for var in t_vars if var.name.startswith('critic_target')]
        if soft == False:
            actor_replace = [tf.assign(at, al) for at,al in zip(at_vars, al_vars)]
            critic_replace = [tf.assign(ct, cl) for ct,cl in zip(ct_vars, cl_vars)]
        else:
            actor_replace = [tf.assign(at, (1-self.TAU)*at + self.TAU * al)
                                  for at, al in zip(at_vars, al_vars)]
            critic_replace = [tf.assign(ct, (1-self.TAU)*ct + self.TAU * cl)
                                   for ct, cl in zip(ct_vars, cl_vars)]
        """
        if soft == False:
            TAU = 1
        else:
            TAU = self.TAU
        al_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'actor_local')
        at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'actor_target')
        cl_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'critic_local')
        ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'critic_target')
        soft_replace = [[tf.assign(ta, (1-TAU)*ta +TAU*la), tf.assign(tc, (1-TAU)*tc + TAU*lc)]
                         for ta,la,tc,lc in zip(at_params, al_params, ct_params, cl_params)]
        return soft_replace
        
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.last_state = next_state
            return experiences
        self.last_state = next_state
        return None

    def random_sample(self, size):
        state = self.reset_episode()
        action_max = self.task.action_high
        action_min = self.task.action_low
        action_range = action_max - action_min
        for i in range(size):
            action = [random.gauss(0, action_range/2) for i in range(self.action_size)]
            next_state, reward, done = self.task.step(action)
            if done:
                next_state = self.reset_episode()
            else:
                pass
            self.step(action, reward, next_state, done)

    def train(self, load_file = None):
        rewards_cur = deque(maxlen=200)
        rewards_avg = []
        a_loss_record = []
        c_loss_record = []
        self.random_sample(size = self.batch_size)

        # build model
        in_state, in_action, in_Qs, in_actions_gradient = self.model_inputs()
        _ , al_loss = Actor(self.task, in_state, in_actions_gradient, self.hidden_size, 'actor_local')
        target_action, at_loss = Actor(self.task, in_state, in_actions_gradient, self.hidden_size, 'actor_target')
        _ , _ , cl_loss = Critic(self.task, in_state, in_action, in_Qs, self.hidden_size, 'critic_local')
        target_value, target_actions_gradient, ct_loss = Critic(self.task, in_state, in_action, in_Qs, self.hidden_size, 'critic_target')
        a_opt, c_opt = self.model_opt(al_loss, cl_loss)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if load_file != None:
                saver.restore(sess, load_file)
            soft_replace = self.update_target(soft = False)
            sess.run(soft_replace)
            print_step = 0
            for i in range(self.num_episodes):
                timestep = 0
                state = self.reset_episode()
                done = False
                t1 = time.time()
                while timestep <= self.runtime or done == False:
                    timestep += 1
                    print_step += 1
                    self.last_state = np.reshape(self.last_state, (1,-1))
                    action = sess.run(target_action, feed_dict={in_state:self.last_state})
                    # add noise
                    action = np.squeeze(action + self.noise.sample()).tolist()
                    next_state, reward, done = self.task.step(action)
                    rewards_cur.append(reward)
                    rewards_avg = np.average(rewards_cur)
                    
                    if done==True:
                        experiences = self.step(action, reward, self.reset_episode(), done)
                    else:
                        experiences = self.step(action, reward, next_state, done)
                    state = next_state
                    if experiences != None:
                        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
                        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.state_size)
                        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
                        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
                        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
                        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.state_size)
                        actions_next = sess.run(target_action, feed_dict = {in_state: next_states})
                        Q_targets_next = sess.run(target_value, feed_dict = {in_state: next_states,
                                                                             in_action: actions_next})

                        Q_targets = rewards + self.gamma * Q_targets_next * (1.0 - dones)
                        (tar_value , tar_actions_gradient)= sess.run([target_value,target_actions_gradient],
                                                            feed_dict = {in_action:actions,
                                                                         in_state:states,
                                                                         in_Qs:Q_targets})
                        _ = sess.run(a_opt,feed_dict = {in_state:states,
                                                        in_actions_gradient:tar_actions_gradient[0],
                                                        in_action:actions})
                        _ = sess.run(c_opt,feed_dict = {in_action:actions,
                                                        in_state:states,
                                                        in_Qs: Q_targets})
                        # Soft-update target models
                        soft_replace = self.update_target(True)
                        sess.run(soft_replace)
                        
                        train_a_loss = at_loss.eval({in_state:states,
                                                     in_actions_gradient:tar_actions_gradient[0],
                                                     in_action:actions})
                        a_loss_record.append(train_a_loss)
                        train_c_loss = ct_loss.eval({in_state:states,
                                                     in_action: actions,
                                                     in_Qs: Q_targets})
                        c_loss_record.append(train_c_loss)
                        print('epoch:{}, avg reward:{}, a loss:{}, c loss:{}'.format(i+1, rewards_avg, train_a_loss, train_c_loss))
                        
                        
                    saver.save(sess,'.checkpoint/actor_critic.ckpt',global_step = i)
                print("epoch total used time:{}".format(time.time()-t1))
