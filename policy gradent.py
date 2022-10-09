
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
import tensorlayer as tl



class buffer(): 
  def __init__(self):
    self.buffer = []

  def new(self):
    self.buffer = []
  
  def add(self,state,action,reward):
    self.buffer.append([state,action,reward])
  
  def replay(self):
    states = []
    actions = []
    dis_rewards = np.zeros(len(self.buffer))
    last_reward = 0
    for i in range(len(self.buffer)):
      states.append(self.buffer[i][0])
      actions.append(self.buffer[i][1])
      dis_rewards[-(i+1)] = last_reward*0.9 + self.buffer[-(i+1)][2]
      last_reward = dis_rewards[-(i+1)]

    dis_rewards -= np.mean(dis_rewards)
    dis_rewards /= np.std(dis_rewards)

    return dis_rewards,states,actions

class Agent():
  def __init__(self,env):
    self.env = env
    self.action_dim = env.action_space.n
    self.state_dim = env.observation_space.shape[0]

    self.model = self.create_model()
    self.model_opti = tf.optimizers.Adam(lr = 0.01)

    self.buffer = buffer()

  
  def create_model(self):
    input = Input(self.state_dim)
    layer1 = Dense(30,activation='relu')(input)
    layer2 = Dense(30,activation='relu')(layer1)
    output = Dense(self.action_dim)(layer2)

    return tf.keras.models.Model(inputs = input, outputs = output)

  def choose_action_pi(self,state):
    prob = self.model(np.array([state],np.float32))
    prob = tf.nn.softmax(prob).numpy()
    return tl.rein.choice_action_by_probs(prob.ravel())


  def choose_action_greedy(self,state):
    state = np.array(state).reshape([1,4])
    action = np.argmax(self.model(state)[0])

    return action

  def train(self):
    
    for i in range(2000):
      state = self.env.reset()
      self.buffer.new()
      done = False
      total_reward = 0
      while done != True:
        action = self.choose_action_pi(state)
        next_state,reward,done,_ = self.env.step(action)
        if done == True:
          reward = -20
        total_reward += reward
        self.buffer.add(state,action,reward)
        state = next_state
        if total_reward >= 2000:
          print(total_reward)
          break
      
      print(i,total_reward)
      dis_rs ,states_buffer, actions_buffer= self.buffer.replay()
    
      with tf.GradientTape() as tape:
        states_buffer = np.array(states_buffer).reshape(-1,4)
        pred = self.model(states_buffer)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred,labels = actions_buffer)*dis_rs)
      
      grad = tape.gradient(loss,self.model.trainable_weights)
      self.model_opti.apply_gradients(zip(grad,self.model.trainable_weights))

    for i in range(20):
      total_reward = 0
      state = self.env.reset()
      done = False
      while done != True:
        self.env.render()
        action = self.choose_action_greedy(state)
        next_state, reward,done,_ = self.env.step(action)
        state = next_state
        total_reward += reward

      print('Reward:',total_reward)
  

if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  env = env.unwrapped
  agent= Agent(env)
  agent.train()
  env.close()
'''

import argparse
import os
import re
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'CartPole-v1'    # 定义环境
RANDOMSEED = 1              # 设置随机种子。建议大家都设置，这样试验可以重现。

DISPLAY_REWARD_THRESHOLD = 400  # 如果奖励超过DISPLAY_REWARD_THRESHOLD，就开始渲染
RENDER = False                  # 开始的时候，不渲染游戏。
num_episodes = 200                # 游戏迭代次数

###############################  PG  ####################################


class PolicyGradient:
    """
    PG class
    """

    def __init__(self, n_features, n_actions, learning_rate=0.01, reward_decay=0.95):
        # 定义相关参数
        self.n_actions = n_actions      #动作
        self.n_features = n_features    #环境特征数量
        self.lr = learning_rate         #学习率
        self.gamma = reward_decay       #折扣

        #用于保存每个ep的数据。
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        def get_model(inputs_shape):
            """
            创建一个神经网络
            输入: state
            输出: act
            """
            
            self.tf_obs = tl.layers.Input(inputs_shape, tf.float32)
                #self.tf_acts = tl.layers.Input([None,], tf.int32, name="actions_num")
                #self.tf_vt = tl.layers.Input([None,], tf.float32, name="actions_value")
            # fc1
            layer = tl.layers.Dense(
                n_units=30, act=tf.nn.tanh, name='fc1')(self.tf_obs)
            # fc2
            all_act = tl.layers.Dense(
                n_units=self.n_actions )(layer)
            return tl.models.Model(inputs=self.tf_obs, outputs=all_act)

        self.model = get_model([None, n_features])
        self.model.train()
        self.optimizer = tf.optimizers.Adam(self.lr)

    def choose_action(self, s):
        """
        用神经网络输出的**策略pi**，选择动作。
        输入: state
        输出: act
        """
        _logits = self.model(np.array([s], np.float32))     
        _probs = tf.nn.softmax(_logits).numpy()             
        return tl.rein.choice_action_by_probs(_probs.ravel())   #根据策略PI选择动作。

    def choose_action_greedy(self, s):
        """
        贪心算法：直接用概率最大的动作
        输入: state
        输出: act
        """
        _logits = self.model(np.array([s]))
        return np.argmax(_logits)

    def store_transition(self, s, a, r):
        """
        保存数据到buffer中
        """
        self.ep_obs.append(s)
        self.ep_as.append(a) 
        self.ep_rs.append(r)

    def learn(self):
        """
        通过带权重更新方法更新神经网络
        """
        # _discount_and_norm_rewards中存储的就是这一ep中，每个状态的G值。
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        with tf.GradientTape() as tape:
            
            # 把s放入神经网络，就算_logits
            self.ep_obs = np.array(self.ep_obs).reshape([-1,4])
            _logits = self.model(self.ep_obs)
            
            # 敲黑板
            ## _logits和真正的动作的差距
            # 差距也可以这样算,和sparse_softmax_cross_entropy_with_logits等价的:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=np.array(self.ep_as))

            # 在原来的差距乘以G值，也就是以G值作为更新
            loss = tf.reduce_mean(neg_log_prob * discounted_ep_rs_norm)

        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data


    def _discount_and_norm_rewards(self):
        """
        通过回溯计算G值
        """
        # 先创建一个数组，大小和ep_rs一样。ep_rs记录的是每个状态的收获r。
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 从ep_rs的最后往前，逐个计算G
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 归一化G值。
        # 我们希望G值有正有负，这样比较容易学习。
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/pg_policy.hdf5', self.model)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/pg_policy.hdf5', self.model)


if __name__ == '__main__':

    # reproducible

    env = gym.make(ENV_NAME)
    env = env.unwrapped

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.9,
    )

    if args.train:

        #=====开始更新训练=====
        for i_episode in range(num_episodes):

            observation = env.reset()
            total_reward = 0
            while True:

              # 注意：这里没有用贪婪算法，而是根据pi随机动作，以保证一定的探索性。
              action = RL.choose_action(observation)
              observation_, reward, done, info = env.step(action)
              if done ==True:
                reward = -20.
              total_reward += reward
              # 保存数o据
              RL.store_transition(observation, action, reward)

              
              # PG用的是MC，如果到了最终状态
              if done:
                ep_rs_sum = sum(RL.ep_rs)

                print(
                        "Episode [%d/%d] \tsum reward: %d " %
                        (i_episode, num_episodes, total_reward)
                    )

                # 开始学习
                RL.learn()

                break
                
                # 开始新一步
              if total_reward >= 2000:
                break

              observation = observation_
        
        RL.save_ckpt()

    # =====test=====
    RL.load_ckpt()
    for i in range(20):
      observation = env.reset()
      done = 0
      total_reward = 0
    
      while done != True:
          env.render()
          action = RL.choose_action_greedy(observation)      # 这里建议大家可以改贪婪算法获取动作，对比效果是否有不同。
          observation, reward, done, info = env.step(action)
          total_reward += reward
      print(total_reward)

'''