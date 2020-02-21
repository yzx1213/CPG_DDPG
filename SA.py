import random
import numpy as np
from lyl_oscillator import oscillator_nw
import math
import matplotlib.pyplot as plt
import vrep


# Initial parameters of the Simulated Annealing Algorithm
T0 = 1000  #initial temperature, the bigger, the better result
t = 0  # evaluation time, used to gradually reduce T0 until convergence
T_end = 0.1  # the end temperature
Markov_num = 50 # the number of markov process, internal iteration

# initialize the optimized parameters in CPG 
# 1. the bound of every parameter
FLT_MIN_KF,    FLT_MAX_KF    = 0.2, 1.0
FLT_MIN_GAIN1, FLT_MAX_GAIN1 = 0.01, 1.0
FLT_MIN_GAIN2, FLT_MAX_GAIN2 = 0.01, 1.0
FLT_MIN_GAIN3, FLT_MAX_GAIN3 = 0.01, 1.0
FLT_MIN_GAIN4, FLT_MAX_GAIN4 = 0.01, 1.0
FLT_MIN_GAIN5, FLT_MAX_GAIN5 = 0.01, 1.0
FLT_MIN_GAIN6, FLT_MAX_GAIN6 = 0.01, 1.0
FLT_MIN_BIAS1, FLT_MAX_BIAS1 = -0.6, 0.0
FLT_MIN_BIAS2, FLT_MAX_BIAS2 = 0.0, 0.5
FLT_MIN_BIAS3, FLT_MAX_BIAS3 = -0.5, 0.0
FLT_MIN_BIAS4, FLT_MAX_BIAS4 = 0.0, 1.0
FLT_MIN_K, FLT_MAX_K = -2.5, 2.5

# 1.2 generate the list of the bound (for the main loop use)
gain_bound_list = [[0.01,1.0]]*6
bound_list = list([[0.2,1.0]])
bound_list.extend(gain_bound_list)
bound_list.extend([[-0.6,0.0],[0.0,0.5],[-0.5,0.0],[0.0,1.0],[-2.5,2.5]])
    
# 2. randomly choose the parameter as the first solution
kf = random.uniform(FLT_MIN_KF, FLT_MAX_KF)
gain1 = random.uniform(FLT_MIN_GAIN1, FLT_MAX_GAIN1)
gain2 = random.uniform(FLT_MIN_GAIN2, FLT_MAX_GAIN2)
gain3 = random.uniform(FLT_MIN_GAIN3, FLT_MAX_GAIN3)
gain4 = random.uniform(FLT_MIN_GAIN4, FLT_MAX_GAIN4)
gain5 = random.uniform(FLT_MIN_GAIN5, FLT_MAX_GAIN5)
gain6 = random.uniform(FLT_MIN_GAIN6, FLT_MAX_GAIN6)
bias1 = random.uniform(FLT_MIN_BIAS1, FLT_MAX_BIAS1)
bias2 = random.uniform(FLT_MIN_BIAS2, FLT_MAX_BIAS2)
bias3 = random.uniform(FLT_MIN_BIAS3, FLT_MAX_BIAS3)
bias4 = random.uniform(FLT_MIN_BIAS4, FLT_MAX_BIAS4)
k = random.uniform(FLT_MIN_K, FLT_MAX_K)

# 2 plus. there is a solution which can make NAO to walk for 2 steps, the parameters are as follows:

kf = 0.2098797258	
gain1 = 0.4083754801
gain1= 0.6
gain2 = 0.4646833695
# gain2 = 0.8	
gain3 = 0.0545742201
# gain3 = 0.1	
gain4 = 0.0197344836    
gain5 = 0.5104797402
gain6 = 0.523562905	
bias1 = -0.0636451512  #-0.349
# bias1 = -0.05	
bias2 = 0.148065662  #+0.698	
bias3 = -0.0410920591#-0.349	
bias4 = 0.0405600582+1.57	
k = 1.7334593213

kf = 0.2098797258	   # control the frequency
gain1 = 0.4083754801   # hip
gain1= 0.6 
gain2 = 0.4646833695   # ankle
gain3 = 0.0545742201   # knee
gain4 = 0.0197344836   # hip_x
gain5 = 0.5104797402   # ankle_x
gain6 = 0.523562905	   # shoulder
bias1 = -0.0636451512  # hip
bias2 = 0.148065662    # knee	
bias3 = -0.0410920591  # ankle	
bias4 = 1.6105600582   # shoulder (the initial state is 1.57)	
k = 1.7334593213       # feedback weight para, optimized with kf



# vector of the parameters
#para_vec = [kf, gain1, gain2, gain3, gain4,gain5, gain6, bias1, bias2, bias3, bias4, k]
#para_vec = [0.2098797258,0.4083754801,0.4646833695,0.0545742201,0.0197344836,0.5104797402,0.523562905,-0.0636451512,0.148065662,-0.0410920591,0.0405600582,1.7334593213] # the trained parameters
para_vec = [kf,gain1,gain2,gain3,gain4,gain5,gain6,bias1,bias2,bias3,bias4,k] # the trained parameters
# 3. judge function for the soulution change 
def judge(dE,T):
    """判断函数，dE<0，接受新的解(优化的参数)，dE>0时，以Metropolis准则接受新的解(确保探索性)
       dE代表内能的变化量，目标使内能逐渐减小,t代表时间"""

    if dE < 0:
        return 1  #返回判断条件True
    else:
        p1 = math.exp(-dE/T)  #Metropolis准则概率
        p2 = random.uniform(0,1) #产生0-1的随机数
        if p1>p2 :
            return 1
        else:
            return 0 

# 4. main loop 
# setting of the bound for the paramters 
delta_kfg_min, delta_kfg_max = -0.01, 0.01  # the bound for kf , gains and bias4
delta_b_min , delta_b_max = -0.005, 0.005   # the bound for bias1 to bias3
delta_k_min , delta_k_max = -0.05, 0.05     # the bound for k

counter =0  # the counter for the while loop, also the time
para_vec_old = para_vec
para_vec_new = para_vec

# the list for plotting 
# plt.ion()  #开启interactive mode 成功的关键函数
# plt.figure(1)
# plot_t = [] # the iteration time (x-axis)
# fitness = [] # the optimized function (y-axis)
while T0 > T_end:
    for i in range(Markov_num):
        # caculate the fitness of the initial soulution
        obj_f_old = oscillator_nw(para_vec_old)  # old (initial) object function 
        print('!!!')
        
        # generate a new solution in the neighboorhood of the para_vec by transform function
        # as the bounds of the parameters are different, the change of the soulutions are different
        # and they are calculated in order
        
        # parameter change 1 : kf, gains and bias4
        delta_kfg = random.uniform(delta_kfg_min, delta_kfg_max)

        for j in range(7):
            para_vec_new[j] += delta_kfg
            # to prevent exceeding the bound
            if para_vec_new[j] < bound_list[j][0] or para_vec_new[j] >bound_list[j][1]:
                para_vec_new[j] -= 2* delta_kfg 
        # change the bias4
        para_vec_new[-2] += delta_kfg
        if para_vec_new[-2] < bound_list[-2][0] or para_vec_new[-2] >bound_list[-2][1]:
            para_vec_new[-2] -= 2 * delta_kfg 

        # parameter change2 : bias1 to bias3
        delta_b = random.uniform(delta_b_min, delta_b_max)
        for j in range(7,10):
            para_vec_new[j] += delta_b
            # to prevent exceeding the bound
            if para_vec_new[j]<bound_list[j][0] or para_vec_new[j] >bound_list[j][1]:
                para_vec_new[j] -= 2* delta_b 
        
        # parameter change3 : k 
        delta_k = random.uniform(delta_k_min, delta_k_max)

        para_vec_new[-1] += delta_k
        if para_vec_new[-1]<bound_list[-1][0] or para_vec_new[-1] >bound_list[-1][1]:
            para_vec_new[-1] -= 2*delta_k
        
        # the new value of the object function 
        obj_f_new = oscillator_nw(para_vec_new)
        
        # judge the  the new and old object function values, to maximize the object function, use old - new
        dE = obj_f_old - obj_f_new
        flag = judge(dE,T0)
        # replace the old parameters by new parameters
        if flag:
            para_vec_old = para_vec_new
        fitness.append(oscillator_nw(para_vec_old))
        #save the parameters and the object function
        file_to_save = []
        file_to_save.extend(para_vec_old)
        file_to_save.append(oscillator_nw(para_vec_old))
        if j == Markov_num -1:
            with open(r'parameters.csv', "a", encoding='utf-8-sig', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # columns_name
                    # writer.writerow(["LHipPitch3", "RHipPitch3", "LKneePitch3","RKneePitch3","LAnklePitch3","LAnkleRoll3","Vx","Vy","Vz"])
                    # Writing to every row
                    writer.writerow(file_to_save)
    
    counter += 1

    # plot in each iteration 
    # plt.plot(plot_t, fitness,'-r')
    # plt.draw()#注意此函数需要调用
    # time.sleep(0.01)

    # use linear method to accelerate in the beginning search
    if counter <= 200:
        T0 = T0/(1+counter)
    else :
        T0 = T0/math.log10(1+counter)
    # Termination 
    if counter >= 2000:
        break
    
    

        

