from os import stat
import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
import tensorlayer as tl

batch_size = 32

class buffer:
    def __init__(self):
        self.capicity = 10000
        self.buffer = []
        self.position = 0

    def push(self,state,action,reward,next_state,done):
        if len(self.buffer) < 10000:
            self.buffer.append(None)   
        self.buffer[self.position] = (state,action,reward,next_state,done)
        self.position = int((self.position+1)%self.capicity)

    
    def sample(self,batch_size):
    
        batch = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, done= map(np.stack, zip(*batch))
        return state, action, reward, next_state, done



class Agent:
    def __init__(self,env):
        
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_range = self.env.action_space.high
        self.var = 2

        self.critic_model = self.create_critic_network([None,self.state_dim],[None,self.action_dim])
        self.critic_target = self.create_critic_network([None,self.state_dim],[None,self.action_dim])
        self.actor_model = self.create_actor_network([None,self.state_dim])
        self.actor_target = self.create_actor_network([None,self.state_dim])
        
        def target_copy(model_w,target_w):
            for weights,target_weights in zip(model_w, target_w):
                target_weights.assign(weights)
        
        target_copy(self.critic_model.trainable_weights,self.critic_target.trainable_weights)
        target_copy(self.actor_model.trainable_weights,self.actor_target.trainable_weights)
        self.actor_model.train()
        self.critic_model.train()
        self.actor_target.eval()
        self.critic_target.eval()

        self.critic_model_optim = tf.optimizers.Adam(lr = 0.002)
        self.actor_model_opti = tf.optimizers.Adam(lr = 0.001)

        self.ema = tf.train.ExponentialMovingAverage(decay=0.99)

        self.buffer = buffer()
    
    def create_actor_network(self,input_shape):
        input_layer = tl.layers.Input(input_shape)
        layer1 = tl.layers.Dense(n_units=64, act=tf.nn.relu)(input_layer)
        layer2 = tl.layers.Dense(n_units=64, act=tf.nn.relu)(layer1)
        layer3 = tl.layers.Dense(n_units=self.action_dim, act=tf.nn.tanh)(layer2)
        layer = tl.layers.Lambda(lambda x: self.action_range * x)(layer3)
        return tl.models.Model(inputs=input_layer, outputs=layer)
    
    def create_critic_network(self,input_state_shape,input_action_shape):
        state_input = tl.layers.Input(input_state_shape)
        action_input = tl.layers.Input(input_action_shape)
        layer1 = tl.layers.Concat(1)([state_input,action_input])
        layer2 = tl.layers.Dense(64,act='relu')(layer1)
        layer3 = tl.layers.Dense(64,act='relu')(layer2)
        output_layer = tl.layers.Dense(1)(layer3)
        return tl.models.Model([state_input,action_input],output_layer)

    def learn(self):
        states,actions,rewards,next_states,done = self.buffer.sample(batch_size)
        rewards = rewards[:, np.newaxis]
        done = done[:, np.newaxis]

        with tf.GradientTape() as tape1:
            pred_a = self.actor_target(next_states)
            pred_q = self.critic_target([next_states,pred_a])
            y= rewards+(1-done)*0.9*pred_q
            pred = self.critic_model([states,actions])
            loss = tf.reduce_mean(tf.square(y-pred))
        
        grads = tape1.gradient(loss,self.critic_model.trainable_weights)
        self.critic_model_optim.apply_gradients(zip(grads,self.critic_model.trainable_weights))
        
        #actor采取不同的动作会得到不同的Q值，Q值越大越倾向于选择该动作，
        # 因此Q越大损失越小，Q越小损失越大，Loss=-mean(Q)
        with tf.GradientTape() as tape2:
            pred_a = self.actor_model(states)
            pred_q = self.critic_model([states,pred_a])
            loss = -tf.reduce_mean(pred_q)
        grads = tape2.gradient(loss,self.actor_model.trainable_weights)
        self.actor_model_opti.apply_gradients(zip(grads,self.actor_model.trainable_weights))

        self.para_update()

    def para_update(self):
        paras = self.actor_model.trainable_weights + self.critic_model.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))
        
    def get_action(self, state, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        """
        self.var *= 0.995 
        action = self.actor_model(np.array([state]))[0]
        if greedy:
            return action
        return np.clip(
            np.random.normal(action, self.var), -self.action_range, self.action_range
        ).astype(np.float32)  # add randomness to action selection for exploration
        
    def train(self):
            
        for i in range(200):
            state = self.env.reset().astype(np.float32)
            total_reward = 0
            done = False
            step = 0
            while done != True:
                action = self.get_action(state)
                next_state,reward,done,_ = self.env.step(action)
                next_state = np.array(next_state,dtype=np.float32)
                done = 1 if done is True else 0
                self.buffer.push(state,action,reward,next_state,done)

                if len(self.buffer.buffer) > batch_size:
                    self.learn()
                
                state = next_state
                total_reward += reward
                step += 1
                if step >= 200:
                    break
            print("Episode:%d, Reward:%f"%(i,total_reward))
    
    def test(self):
        self.critic_model.load_weights(r'C:\Users\19758\Desktop\model\critic.h5')
        self.critic_target.load_weights(r'C:\Users\19758\Desktop\model\critic_target.h5')
        self.actor_model.load_weights(r'C:\Users\19758\Desktop\model\actor.h5')
        self.actor_target.load_weights(r'C:\Users\19758\Desktop\model\actor_target.h5')
        for i in range(20):
            state = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            while done != True:
                self.env.render()
                action = self.get_action(state,greedy=True)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                total_reward += reward
                step += 1
                if step > 200:
                    break
            print("Episode:%d, Reward:%f"%(i,total_reward))
           


if __name__ == "__main__":
    env = gym.make('Pendulum-v1').unwrapped
    agent = Agent(env)
    agent.train()
    env.close()