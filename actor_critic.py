'''

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import random
import gym



class Agent():
    def __init__(self,env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.actor_model = self.create_network_actor([None,self.state_dim])
        self.critic_model = self.create_network_critic([None,self.state_dim])
        self.actor_model.train()
        self.critic_model.train()

        self.opti_actor = tf.optimizers.Adam(lr = 0.0005)
        self.opti_critic = tf.optimizers.Adam(lr = 0.001)

    def create_network_actor(self,shape):
        input_layer = tl.layers.Input(shape)
        layer1 = tl.layers.Dense(20,act='relu')(input_layer)
        output_layer = tl.layers.Dense(self.action_dim)(layer1)
        return tl.models.Model(inputs=input_layer, outputs=output_layer)
    
    def create_network_critic(self,shape):
        input_layer = tl.layers.Input(shape)
        layer1 = tl.layers.Dense(32,act='relu')(input_layer)
        output_layer = tl.layers.Dense(1)(layer1)
        return tl.models.Model(inputs=input_layer,outputs=output_layer)
    
    def choose_action_pi(self,state):
        pred = self.actor_model(np.array([state]))[0]
        pred = tf.nn.softmax(pred).numpy()
        return tl.rein.choice_action_by_probs(pred.ravel())

    def get_td_error(self,state,next_state,reward,done):
        with tf.GradientTape() as tape1:
            d = 0 if done else 1
            td_error = reward+0.9*d*self.critic_model(np.array([next_state]))[0] - self.critic_model(np.array([state]))[0]
            loss =  tf.square(td_error)
        grads = tape1.gradient(loss,self.critic_model.trainable_weights)
        self.opti_critic.apply_gradients(zip(grads,self.critic_model.trainable_weights))

        return td_error

    def learn(self,td_error,state,action):
    
        with tf.GradientTape() as tape2:
            pred = self.actor_model(np.array([state]))[0]
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=action)
            loss = loss*td_error
        grads = tape2.gradient(loss,self.actor_model.trainable_weights)
        self.opti_actor.apply_gradients(zip(grads,self.actor_model.trainable_weights))
        
    def epsilon_greedy(self,state):
        if random.random() <=0.15:
            return random.choice(range(self.action_dim))
        else:
            pred = self.actor_model(np.array([state]))[0]
            return np.argmax(pred)

    def train(self):
        for i in range(500):
            state = self.env.reset()
            done = False
            total_reward = 0
            while done != True:
                action = self.epsilon_greedy(state)
                next_state,reward,done,_ = self.env.step(action)
                if done:
                    reward = -20.
                total_reward += reward
                
                td_error = self.get_td_error(state,next_state,reward,done)
                self.learn(td_error,state,action)

                state = next_state
                
                if total_reward >=2000:
                    print('Episode:%d , Reward:%f'%(i,total_reward))
                    break
            
            print('Episode:%d , Reward:%f'%(i,total_reward))
        


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.unwrapped
    agent = Agent(env)
    agent.train()
    env.close()
'''

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# 超参数
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # 刷新阈值
MAX_EP_STEPS = 1000   		    #最大迭代次数
RENDER = False  # 渲染开关
GAMMA = 0.9     # 衰变值
LR_A = 0.001    # Actor学习率
LR_C = 0.01     # Critic学习率

env = gym.make('CartPole-v0')
env.seed(1)  
env = env.unwrapped

N_F = env.observation_space.shape[0] # 状态空间
N_A = env.action_space.n		     # 动作空间


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # 获取所有操作的概率
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
 
sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)  # 初始化Actor
critic = Critic(sess, n_features=N_F, lr=LR_C)     # 初始化Critic
sess.run(tf.global_variables_initializer())        # 初始化参数

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)        # 输出日志

# 开始迭代过程 对应伪代码部分
for i_episode in range(MAX_EPISODE):
    s = env.reset() # 环境初始化
    t = 0
    track_r = []    # 每回合的所有奖励
    total_reward = 0
    while True:
        if RENDER: env.render()
        a = actor.choose_action(s)       # Actor选取动作
        s_, r, done, info = env.step(a)   # 环境反馈
        if done: r = -20    # 回合结束的惩罚
        total_reward += r
        track_r.append(r)  # 记录回报值r
        td_error = critic.learn(s, r, s_)  # Critic 学习
        actor.learn(s, a, td_error)        # Actor 学习
        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            # 回合结束, 打印回合累积奖励
            ep_rs_sum = sum(track_r)
        
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = False  # rendering
            print("episode:", i_episode, "  reward:", int(total_reward))
            break
