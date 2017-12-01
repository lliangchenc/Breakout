import numpy as np
import tensorflow as tf
import random
from collections import deque 

#global parameters
EPSILON = 0.1
WEIGHT_DECAY = 0.95
MAX_MEMORY_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.9
UPDATE_FREQ = 100

class DQN:
    def __init__(self, num_actions):
    
        #evaluation Q_Network
        self.train_deque = deque()
        self.step = 0;
        self.current_state = []
        self.epsilon = EPSILON
        self.actions = num_actions
        self.QValue, self.W_conv1, self.W_conv2, self.W_conv3, self.W_fc1, self.W_fc2, \
        self.bias_conv1, self.bias_conv2, self.bias_conv3, self.bias_fc1, self.bias_fc2, self.QInput =  self.QNetWork()

        #target Q_Network
        self.QValue_T, self.W_conv1_T, self.W_conv2_T, self.W_conv3_T, self.W_fc1_T, self.W_fc2_T, \
        self.bias_conv1_T, self.bias_conv2_T, self.bias_conv3_T, self.bias_fc1_T, self.bias_fc2_T, self.QInput_T =  self.QNetWork()

        self.action_input = tf.placeholder(tf.float32, [None, self.actions])
        self.y_input = tf.placeholder(tf.float32, [None])

        #Q_learning
        self.Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.action_input), axis = (1))
        self.loss = tf.reduce_mean(tf.square(self.y_input - self.Q_Action))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("train")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print "Model Loaded : ", checkpoint.model_checkpoint_path
        else:
            print "New Model"

    def initState(self, observation):
        self.current_state = np.stack((observation, observation, observation, observation), axis = 2)

    def QNetWork(self):
        input = tf.placeholder(tf.float32, [None, 210, 160, 4])
        W_conv_1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32]))
        W_conv_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64]))
        W_conv_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64]))
        W_fc_1 = tf.Variable(tf.truncated_normal([4 * 3 * 64, 100]))
        W_fc_2 = tf.Variable(tf.truncated_normal([100, self.actions]))

        bias_conv_1 = tf.Variable(tf.truncated_normal([32]))
        bias_conv_2 = tf.Variable(tf.truncated_normal([64]))
        bias_conv_3 = tf.Variable(tf.truncated_normal([64]))
        bias_fc_1 = tf.Variable(tf.truncated_normal([100]))
        bias_fc_2 = tf.Variable(tf.truncated_normal([self.actions]))

        conv1 = tf.nn.conv2d(input, W_conv_1, strides = [1,4,4,1], padding = "SAME")
        conv1 = tf.nn.relu(conv1 + bias_conv_1)
        pooling1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

        conv2 = tf.nn.conv2d(pooling1, W_conv_2, strides = [1,2,2,1], padding = "SAME")
        conv2 = tf.nn.relu(conv2 + bias_conv_2)
        pooling2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

        conv3 = tf.nn.conv2d(pooling2, W_conv_3, strides = [1,2,2,1], padding = "SAME")
        conv3 = tf.nn.relu(conv3 + bias_conv_3)
        
        fc1 = tf.nn.relu(tf.matmul(tf.reshape(conv3,[-1, 4 * 3 * 64]), W_fc_1) + bias_fc_1)
        Q_value = tf.matmul(fc1, W_fc_2) + bias_fc_2 # n * actions
        return Q_value, W_conv_1, W_conv_2, W_conv_3, W_fc_1, W_fc_2, \
                bias_conv_1, bias_conv_2, bias_conv_3, bias_fc_1, bias_fc_2, input

    def feedFrame(self, observations, action, reward, is_term):
        temp = [self.current_state, action, reward, is_term]
        self.current_state = np.append(self.current_state[:, :, 1:],observations[:, :, np.newaxis], axis=2)
        temp.append(self.current_state)
        self.train_deque.append(temp)
        self.step += 1
        if(len(self.train_deque) >= MAX_MEMORY_SIZE):
            self.train_deque.popleft()
        if(len(self.train_deque) >= BATCH_SIZE):
            self.train()

    
    def getAction(self):

        QValue = self.QValue.eval(feed_dict = {self.QInput : [self.current_state]})[0]
        index = np.argmax(QValue)
        if np.random.rand() < EPSILON:
            index = np.random.randint(self.actions)
            #print "RANDOM"
        temp = np.zeros(self.actions)
        temp[index] += 1
        #print "ACTION", index
        return index

    def train(self):
        batch = random.sample(self.train_deque, BATCH_SIZE)
        current_state_batch = []
        action_batch = []
        reward_batch = []
        is_term_batch = []
        next_state_batch = []

        for i in range(BATCH_SIZE):
            current_state_batch.append(batch[i][0])
            action_batch.append(batch[i][1])
            reward_batch.append(batch[i][2])
            is_term_batch.append(batch[i][3])
            next_state_batch.append(batch[i][4])

        Q_Value_T = self.QValue_T.eval(feed_dict = {self.QInput_T : next_state_batch})
        y_ = []

        for i in range(BATCH_SIZE):
            if is_term_batch[i]:
                y_.append(reward_batch[i])
            else:
                y_.append(reward_batch[i] + GAMMA * np.max(Q_Value_T[i]))

        _, loss = self.session.run([self.optimizer, self.loss], feed_dict = {self.QInput : current_state_batch, self.action_input : action_batch, self.y_input : y_})

        if self.step % 100 == 0:
            print self.step, loss
        if self.step % 10000 == 0:
            self.saver.save(self.session, 'train/' + 'DQN', global_step = self.step)
        
        if self.step % UPDATE_FREQ == 0:
            self.epsilon = max(self.epsilon * WEIGHT_DECAY, 0.001)
            self.session.run([self.W_conv1_T.assign(self.W_conv1), self.W_conv2_T.assign(self.W_conv2), self.W_conv3_T.assign(self.W_conv3), \
                           self.bias_conv1_T.assign(self.bias_conv1), self.bias_conv2_T.assign(self.bias_conv2), self.bias_conv3_T.assign(self.bias_conv3),\
                           self.W_fc1_T.assign(self.W_fc1), self.W_fc2_T.assign(self.W_fc2), self.bias_fc1_T.assign(self.bias_fc1), self.bias_fc2_T.assign(self.bias_fc2)])   
