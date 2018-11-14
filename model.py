import tensorflow as tf
import numpy as np
def Actor(task, state, actions_gradient, hidden_size, scope = 'Actor', training = True):
    action_max = task.action_high
    action_min = task.action_low
    action_range = action_max - action_min
    with tf.variable_scope(scope,reuse = False):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        fc1 = tf.layers.dense(state, hidden_size, 
                              activation = None,
                              kernel_regularizer=regularizer)
        fc1 = tf.nn.leaky_relu(fc1, alpha=0.2)
        fc2 = tf.layers.dense(fc1, hidden_size*2, 
                              activation = None,
                              kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.1),
                              bias_initializer = tf.constant_initializer(0.1),
                              kernel_regularizer=regularizer)
        fc2 = tf.layers.batch_normalization(fc2, training = training)
        fc2 = tf.nn.leaky_relu(fc2, alpha=0.2)
        fc3 = tf.layers.dense(fc2, hidden_size, 
                              activation = None,
                              kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.1),
                              bias_initializer = tf.constant_initializer(0.1),
                              kernel_regularizer=regularizer)
        fc3 = tf.nn.leaky_relu(fc3, alpha=0.2)
        sigmoid_action = tf.layers.dense(fc3, task.action_size, activation=tf.nn.sigmoid)
        action = tf.identity(tf.add(tf.multiply(sigmoid_action, action_range), action_min))
        loss = tf.reduce_mean(tf.multiply(action, actions_gradient)) + tf.losses.get_regularization_loss()
        return action, loss
	
def Critic(task, state, action, Qs_target, hidden_size=32, scope = 'Critic', training = True):
    with tf.variable_scope(scope, reuse = False):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        state_fc = tf.layers.dense(state, hidden_size,
                                   activation = None,
                                   kernel_regularizer=regularizer)
        state_fc = tf.nn.leaky_relu(state_fc, alpha=0.2)
        state_out = tf.layers.dense(state_fc, hidden_size*2,
                                    activation = None,
                                    kernel_regularizer=regularizer)
        state_out = tf.nn.leaky_relu(state_out, alpha=0.2)							   
        action_fc = tf.layers.dense(action, hidden_size,
                                    activation = None,
                                    kernel_regularizer=regularizer)
        action_fc = tf.nn.leaky_relu(action_fc, alpha=0.2)
        action_out = tf.layers.dense(action_fc, hidden_size*2,
                                     activation = None,
                                     kernel_regularizer=regularizer)
        action_out = tf.nn.leaky_relu(action_out, alpha=0.2)								   
        net = tf.concat([state_out, action_out], 1)
        net = tf.layers.dense(net, hidden_size,
                              activation = None,
                              kernel_regularizer=regularizer)
        net = tf.nn.leaky_relu(net, alpha = 0.2)
        value = tf.identity(tf.layers.dense(net, 1,
                                activation = None))

        actions_gradient = tf.gradients(value, action)
        
        loss = tf.reduce_mean(tf.square(Qs_target - value)) + tf.losses.get_regularization_loss()
        return value, actions_gradient, loss
