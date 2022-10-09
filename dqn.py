import random
import numpy as np
import gym
import tensorflow as tf

import tensorlayer as tl

train_episodes = 1000
batch_size = 32
garmma = 0.9
save = []


class replay_buffer:
    def __init__(self):
        self.capacity = 10000
        self.buffer = []
        self.position = 0
        

    def push(self,state,action,reward,next_state,Done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state,action,reward,next_state,Done)
        self.position = int((self.position+1) % self.capacity)
    
    def sample(self,batches_size):
        batch = random.sample(self.buffer,batches_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done 

class Agent:
    def __init__(self,env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.model = self.create_network()
        self.target_model = self.create_network()
        self.model.train()
        self.target_model.eval()
        self.model_optim = tf.optimizers.Adam(lr = 0.005)
        #self.model_traget_optim = tf.optimizers.Adam(lr = 0.01)
        
        self.buffer = replay_buffer()
        self.epsilon = 0.15


    def create_network(self):
        input_layer = tl.layers.Input([None,self.state_dim])
        layer1 = tl.layers.Dense(20,act='relu')(input_layer)
        #layer2 = tl.layers.Dense(16,activation='relu')(layer1)
        output_layer = tl.layers.Dense(self.action_dim)(layer1)
        return tl.models.Model(inputs = input_layer, outputs = output_layer)

    def traget_update(self):
        for weights,target_weights in zip(self.model.trainable_weights, self.target_model.trainable_weights):
            target_weights.assign(weights)

    
    def epsilon_greedy(self,state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_dim))
        else:
            state = np.array(state).reshape([1,4])
            pred = self.model([state])[0]
            return np.argmax(pred)
        
    
    def replay(self):
        for i in range(10):
            states,actions,rewards,next_states,dones = self.buffer.sample(batch_size)
            target = self.target_model(states).numpy()
            next_target = self.target_model(next_states)
            next_q_value = tf.reduce_max(next_target,axis = 1)

            target[range(batch_size),actions] = rewards + (1-dones)*garmma*next_q_value

            with tf.GradientTape() as tape:
                q_pred = self.model(states)
                loss = tf.losses.mean_squared_error(target,q_pred)
            
            grads = tape.gradient(loss,self.model.trainable_weights)
            self.model_optim.apply_gradients(zip(grads,self.model.trainable_weights))
        
        self.traget_update()

    def update_epsilon(self):
        self.epsilon *= 0.999
        if self.epsilon <= 0.05:
            self.epsilon = 0.05

    def training(self):
        for i in range(train_episodes):
            total_reward = 0
            Done = 0
            state = self.env.reset()
            while Done != 1:
                action = self.epsilon_greedy(state)
                #self.update_epsilon()
                next_state,reward,Done,_ = self.env.step(action)
                if Done:
                    reward = -20.
                total_reward += reward
                self.buffer.push(state,action,reward,next_state,Done)
                state = next_state
                
                if total_reward >= 2000:
                    break
            
            if len(self.buffer.buffer) > batch_size:
                    self.replay()



            print('Episode:%d, Reward:%f, Epsilon:%f'%(i,total_reward,self.epsilon))

            
        for i in range(20):
            state_t = self.env.reset()
            Done_t = 0
            total_reward_t = 0
            while Done_t != 1:
                self.env.render()
                state_t = np.array(state_t).reshape([1,4])
                action_t = np.argmax(self.model(state_t)[0])
                next_state_t,reward_t,Done_t,_ = env.step(action_t)
                state_t = next_state_t
                total_reward_t += reward_t
            print('TEST  Reward:%f'%total_reward_t)
            


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    agent = Agent(env)
    agent.training()
    env.close()


