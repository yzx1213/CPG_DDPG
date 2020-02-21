import tensorflow as tf
import numpy as np
import os
import shutil 
import vrep
import time
import math
import matplotlib.pyplot as plt
import random
from CPG_single import oscillator_nw
from DDPG import DDPG
import csv

clientID=vrep.simxStart('127.0.0.1',19997,True,True,-500000,5)
res, HeadHandle = vrep.simxGetObjectHandle(clientID, "HeadYaw", vrep.simx_opmode_blocking)
res, NAOHandle = vrep.simxGetObjectHandle(clientID, "NAO", vrep.simx_opmode_blocking)
res, NAO_Pos0 = vrep.simxGetObjectPosition(clientID,NAOHandle,-1,vrep.simx_opmode_blocking)
x0 = NAO_Pos0[0]
y0 = NAO_Pos0[1]
max_x = float('-inf')


def getState(dx):
    res, NAO_Pos = vrep.simxGetObjectPosition(clientID,NAOHandle,-1,vrep.simx_opmode_blocking)
    res, Orientation = vrep.simxGetObjectOrientation(clientID, NAOHandle, -1, vrep.simx_opmode_blocking)
    # res, NAO_V, NAO_Angular_V = vrep.simxGetObjectVelocity(clientID, NAOHandle, vrep.simx_opmode_blocking)
    #这里的速度与角速度是否可以用上去？
    s = []
    # state中存储绝对坐标中y的偏移量
    dy = NAO_Pos[1]-y0
    s.append(dy)
    #s.append(dx)
    s += Orientation

    isFall = 0
    isOut  = 0
    res, HeadPos = vrep.simxGetObjectPosition(clientID, HeadHandle, -1, vrep.simx_opmode_blocking)
    if HeadPos[2] < 0.42: isFall = 1
    if (abs(dy) > 0.1) or (abs(Orientation[2]) > 0.5): isOut = 1
    return s, isFall, isOut

def CalculateReward(dx,y):#计算reward的方式是否合理？
    w1 = 2
    w2 = 1
    reward = w2*dx - y*w1
    print('dy:',y)
    print('dx:',dx)
    return reward


def step(position_vector):
    global max_x
    res, NAO_Pos1 = vrep.simxGetObjectPosition(clientID, NAOHandle, -1, vrep.simx_opmode_blocking)
    # print(clientID)
    res_ = oscillator.oscillator_step(position_vector)
    i = 0
    ls=[]
    while True:
        res = oscillator.oscillator_step(position_vector)
        ls.append((res - bias1) / gain1)
        vrep.simxSynchronousTrigger(clientID)
        if res * res_ < 0:
            i += 1
            res_=res
        if i==2:
            break
    # for i in range(20): 
    #     oscillator.oscillator_step(position_vector)
    #     vrep.simxSynchronousTrigger(clientID)
    res, NAO_Pos2 = vrep.simxGetObjectPosition(clientID, NAOHandle, -1, vrep.simx_opmode_blocking)
    dx = NAO_Pos2[0] - NAO_Pos1[0]
    dy = NAO_Pos2[1] - NAO_Pos1[1] #相对偏移量
    
    #print('x: ', NAO_Pos2[0]+2)
    #print('y: ', NAO_Pos2[1])
    if NAO_Pos2[0] > max_x:
        max_x = NAO_Pos2[0]
        print('best_x updated!')
    print('best_x: ', max_x+2)

    r = CalculateReward(dx, abs(NAO_Pos2[1]))
    s_, isFall, isOut = getState(dx)
    
    if isFall: r -= 0.1
    if isOut : r -= 0.1
    return s_, r, isFall, isOut, dx, NAO_Pos2[1], NAO_Pos2[0],ls

    



kf = 0.2098797258	   # control the frequency
gain1 = 0.4083754801   # hip
gain1= 0.6 
gain2 = 0.4646833695   # ankle
gain3 = 0.0545742201   # knee
gain3 = 0.04
gain4 = 0.0197344836   # hip_x
gain5 = 0.5104797402   # ankle_x
gain6 = 0.523562905	   # shoulder
bias1 = -0.0636451512  # hip
bias2 = 0.148065662    # knee	
bias3 = -0.0410920591  # ankle	
bias4 = 1.6105600582   # shoulder (the initial state is 1.57)	
k = 1.7334593213       # feedback weight para, optimized with kf

para_vec = [kf, gain1, gain2, gain3, gain4, gain5, gain6, bias1, bias2, bias3, bias4, k]

MAX_EPISODES = 20000
MAX_EP_STEPS = 100
SAVE_MODEL_ITER = 500
MAX_EPISODE_REWARD = -100

STEP_COUNT = 0


# STATE_DIM = 5
STATE_DIM = 4
ACTION_DIM = 8
#ACTION_BOUND = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.01, 0.05]
#ACTION_BOUND = [0.1, 0.3, 0.2, 0.03, 0.03, 0.2, 0.05, 0.005, 0.05, 0.01, 0.05, 0.1]
ACTION_BOUND = [0.01, 0.01, 0.01, 0.01, 0.01,0.005,0.005,0.005] 
var = 0.5
var_min = 0.001

epsilon = 1
epsilon_decay = 0.95
osc_ls=[]

if __name__ == "__main__":
    ddpg = DDPG(ACTION_DIM, STATE_DIM, ACTION_BOUND)
    start = time.time()
    print('start.', start)
    t = time.strftime("%m%d_%H%M%S", time.localtime(time.time()))
    with open("performance_data/{}.csv".format(t), 'a') as f: 
        f.write('i_epi,epi_r,x,epi_step,y_final\n') #一个episode结束的dy
    for i_episode in range(MAX_EPISODES):
        vrep.simxSynchronous(clientID, 1)
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
        #print('#################start##################')
        
        print("Now we are starting a new episode.")
        global oscillator 
        oscillator= oscillator_nw(para_vec, clientID) 
        episode_reward = 0
        x, dx, dy = 0, 0, 0
        for i_step in range(MAX_EP_STEPS):
                print("-------------------------------------------------------------------------")
                s, _ ,_ = getState(dx)
                if np.random.random() > epsilon:
                    a = ddpg.choose_action(np.array(s))   #a is d_para
                    a = np.clip(a, -1*np.array(ACTION_BOUND), np.array(ACTION_BOUND))
                    
                    print('use a.')
                else:
                    a = np.random.normal(np.zeros(ACTION_DIM), var)*np.array(ACTION_BOUND)
                    print('use random.')
                    #print('a:', a)#4
                a_ = [0,a[0],a[1],a[2],a[3],a[4],0,a[5],a[6],a[7],0,0]
                para_vec_in = para_vec+ a_   
                
                s_, r, isFall, isOut, dx, y, x, osc = step(para_vec)
                
                osc_ls+=osc
                episode_reward+=r

                ddpg.store_transition(s, a, r, s_)
                data = []
                data = np.append(data, s)
                data = np.append(data, a)
                data = np.append(data, [r])
                data = np.append(data, s_)
                

                STEP_COUNT += 1
                if STEP_COUNT > 500:
                    
                    print("I'm learning")
                    for i in range(1): ddpg.learn()

                    if STEP_COUNT % SAVE_MODEL_ITER == 0:
                        ddpg.save_model(STEP_COUNT)
                        print('model saved. Used time: ', int(time.time() - start))
                
                if i_episode > 200:
                    if i_episode % 3 == 0:
                        epsilon *= epsilon_decay
                
                if episode_reward > MAX_EPISODE_REWARD:
                        MAX_EPISODE_REWARD = episode_reward

                print('Episode:', i_episode,
                    '| Step: %i' % i_step,
                    '| Epi_reward: %f' % episode_reward,
                    '| Max_epi_r: %f' % MAX_EPISODE_REWARD,     
                    '| Reward for this step: %f' % r,
                    '| Global steps: %i' % STEP_COUNT,
                    '| Epsilon: %.4f' % epsilon,
                    '| ls: ',osc_ls
                )  

                if isFall:
                    print("The robot falls down.")
                    break
                if isOut:
                    print("The robot is outside")
                    break
        with open("performance_data/{}.csv".format(t), 'a') as f: 
            x+=2
            f.write(str(i_episode)+','+str(episode_reward)+','+str(x)+','+str(i_step)+','+str(y)+'\n')
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)

    vrep.simxFinish(clientID)
    print("Finish.")

# if __name__ == "__main__":
#     for i_episode in range(MAX_EPISODES):
#         vrep.simxSynchronous(clientID, 1)
#         vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
#         #print('#################start##################')

#         print("Now we are starting a new episode.")
#         episode_reward = 0
#         #para_vec_old = para_vec
#         dx = 0
#         for i_step in range(MAX_EP_STEPS):
#             print("-------------------------------------------------------------------------")

#             s, _ ,_ = getState(dx)
#             para_vec_in = para_vec
#             s_, r, isFall, isOut, dx = step(para_vec_in)
#             episode_reward+=r

#             data = []
#             data = np.append(data, s)
#             data = np.append(data, [r])
#             data = np.append(data, s_)

#             STEP_COUNT += 1

#             print('Episode:', i_episode,
#                 '| Step: %i' % i_step,
#                 '| Epi_reward: %f' % episode_reward,
#                 '| Exploration: %.3f' % var,
#                 '| Reward for this step: %f' % r,
#                 '| Global steps: %i' % STEP_COUNT,
#             )

#             if isFall:
#                 print("The robot falls down.")
#                 break
#             if isOut:
#                 print("The robot is outside")
#                 break

#         vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
#         time.sleep(1)

#     vrep.simxFinish(clientID)
#     print("Finish.")
