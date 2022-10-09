import gym
import random
import time


grid = gym.make('GridWorld-v0')
states = grid.get_states()
actions = grid.get_actions()

def greedy(q_value,state):
    key = '%d_%s'%(state,actions[0])
    q_max = q_value[key]
    a_max = 0

    for i in range(4):
        key = '%d_%s'%(state,actions[i])
        if q_max < q_value[key]:
            q_max = q_value[key]
            a_max = i

    return q_max,a_max

def epsilon_greedy(q_value,epsilon,state):
    if random.random() > (1-epsilon):
        return random.choice(actions)
    else:
        key = '%d_%s'%(state,actions[0])
        q_max = q_value[key]
        a_max = 0

        for i in range(4):
            key = '%d_%s'%(state,actions[i])
            if q_max < q_value[key]:
                q_max = q_value[key]
                a_max = i
        return actions[a_max]



def q_learning(iter,alpha,epsilon,garma):

    #初始化
    q_value = dict()
    for s in states:
        for a in actions:
            key = '%d_%s'%(s,a)
            q_value[key] = 0.
    
    for i in range(iter):
        state = grid.reset()
        terminate = False

        while terminate == False:
            a = epsilon_greedy(q_value,epsilon,state)
            next_state,reward,terminate,info = grid.step(a)
            key = '%d_%s'%(state,a)
            q_max,amax= greedy(q_value,next_state)
            q_value[key] += alpha*(reward+garma*q_max-q_value[key])
            state = next_state
        
        print('Episode %d is done'%i)

    return q_value

def policy(q_value):
    for s in states[0:17]:
        q_max,a_max = greedy(q_value,s)
        print('At state %d, the best policy is %s'%(s,actions[a_max]))



if __name__ == '__main__':

    q_func = q_learning(500,0.1,0.9,0.8)
    policy(q_func)







