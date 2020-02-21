import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import os
import csv
import time
import vrep 
import random
from Fitness import calc_fitness

sm_t_list=[]
sm_p_list=[]
pointer=0

def oscillator_nw(position_vector):

    # Connect to Vrep 
    print('Program started')
    vrep.simxFinish(-1) # just in case, close all opened connections
    global clientID 
    clientID= vrep.simxStart('127.0.0.1', 19997, True, True, -500000, 5) # Connect to V-REP, set a very large time-out for blocking commands
    if clientID != -1:
        print('Connected to remote API server')
    
        # Save all the joints in a list named Joints
        # vrep.simxSetBooleanParameter(clientID, sim.boolparam_display_enabled, false)
        headJoints = ['HeadYaw', 'HeadPitch']
        leftLegJoints = ['LHipYawPitch3', 'LHipRoll3', 'LHipPitch3', 'LKneePitch3', 'LAnklePitch3', 'LAnkleRoll3']
        rightLegJoints = ['RHipYawPitch3', 'RHipRoll3', 'RHipPitch3', 'RKneePitch3', 'RAnklePitch3', 'RAnkleRoll3']
        leftArmJoints = ['LShoulderPitch3', 'LShoulderRoll3', 'LElbowYaw3', 'LElbowRoll3', 'LWristYaw3']
        leftHand = ['NAO_LThumbBase', 'Revolute_joint8', 'NAO_LLFingerBase', 'Revolute_joint12', 'Revolute_joint14',
                'NAO_LRFingerBase', 'Revolute_joint11', 'Revolute_joint13']
        rightArmJoints = ['RShoulderPitch3', 'RShoulderRoll3', 'RElbowYaw3', 'RElbowRoll3', 'RWristYaw3']
        rightHand = ['NAO_RThumbBase', 'Revolute_joint0', 'NAO_RLFingerBase', 'Revolute_joint5', 'Revolute_joint6',
                 'NAO_RRFingerBase', 'Revolute_joint2', 'Revolute_joint3']
    
                 # Extract the joints controllled
        Ctrl_Joints = ['LHipPitch3','RHipPitch3','LKneePitch3','RKneePitch3','LAnklePitch3','RAnklePitch3','LHipRoll3',
                   'LAnkleRoll3','RHipRoll3','RAnkleRoll3','LShoulderPitch3','RShoulderPitch3']
                   
        #print ("LegJoints:",LegJoints)
    
        #handle the NAO object 
        res, NAOHandle = vrep.simxGetObjectHandle(clientID, "NAO", vrep.simx_opmode_blocking)
    
        #handle the CtrlJoints
        Handle = [None] * len(Ctrl_Joints)
        for i in range (len(Ctrl_Joints)):
            res,Handle[i] = vrep.simxGetObjectHandle(clientID,Ctrl_Joints[i],vrep.simx_opmode_blocking)
        
        res,lHipHandle = vrep.simxGetObjectHandle(clientID,'LHipPitch3',vrep.simx_opmode_blocking)
        res,rHipHandle = vrep.simxGetObjectHandle(clientID,'RHipPitch3',vrep.simx_opmode_blocking)
	#####################  VREP   #######   HANDLE   ########   PART   ##############################


        #define ocilator (Ref:Cristiano 2014 Locomotion Control of a Biped Robot Through a Feedback CPG Network )
        kf=position_vector[0]
        gain1=position_vector[1]
        gain2=position_vector[2]
        gain3=position_vector[3]
        gain4=position_vector[4]
        gain5=position_vector[5]
        gain6=position_vector[6]
        bias1=position_vector[7]
        bias2=position_vector[8]
        bias3=position_vector[9]
        bias4=position_vector[10]
        k=position_vector[11]	

        """Caculate the CPG model and return the fitness for Genetic Algorithm"""
        #Constants in Matsuoka Ocilator Model
        tau = 0.2  # rise time constant,determine the frequency
        tau_prime = 0.4977  # adaptation effect time constant
        beta = 2.5           # self-inhibition constant
        w0 = 2.2829          # constant of mutation inhibition between flexor and extensor
        ue = 0.4111          # tonic input to neurons
        m1 = 1.0             # weights used to calculate output (extensor neuron)
        m2 = 1.0             # weights used to calculate  output (flexor neuron)
        a = 1.0               # constant of interaction between ocillators

        # kf is a constant to modulate the time constant
        tau = tau*kf
        tau_prime = tau_prime * kf
    
        # Define step time
        dt = 0.01   #(50ms)

        # Variables in Matsuoka Ocilator Model (subscript 1 for extensor, 2 for flexor)
    
        # u1i represents the extensor neuron for the ith oscillator, u2i represents the flexor neuron for the ith oscillator
        # u controls the discharge rate, v controls the self-inhibition degree
        # y is the output for one kind of neuron, oi is the final ouput(linear combination of y)
    
        # Oscilator 1 (Pacemaker, drive other oscillators , control no joint)
        u1_1,  u2_1,  v1_1,  v2_1,  y1_1,  y2_1,  o_1,  gain_1,  bias_1  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        # Oscillator 2 (LHipPitch, angle feedback)
        u1_2,  u2_2,  v1_2,  v2_2,  y1_2,  y2_2,  o_2,  gain_2,  bias_2  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain1, bias1
        # Oscillator 3 (RHipPitch,angle feedback)
        u1_3,  u2_3,  v1_3,  v2_3,  y1_3,  y2_3,  o_3,  gain_3,  bias_3  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain1, bias1
        # Oscillator 4 (LKneePitch)
        u1_4,  u2_4,  v1_4,  v2_4,  y1_4,  y2_4,  o_4,  gain_4,  bias_4  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain3, bias2
        # Oscillator 5 (RKneePitch)
        u1_5,  u2_5,  v1_5,  v2_5,  y1_5,  y2_5,  o_5,  gain_5,  bias_5  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain3, bias2
        # Oscillator 6 (LAnklePitch)
        u1_6,  u2_6,  v1_6,  v2_6,  y1_6,  y2_6,  o_6,  gain_6,  bias_6  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain2, bias3
        # Oscillator 7 (RAnklePitch)
        u1_7,  u2_7,  v1_7,  v2_7,  y1_7,  y2_7,  o_7,  gain_7,  bias_7  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain2, bias3
        # Oscillator 8 (LHipRoll)
        u1_8,  u2_8,  v1_8,  v2_8,  y1_8,  y2_8,  o_8,  gain_8,  bias_8  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain4, 0.0
        # Oscillator 9 (lAnkleRoll)
        u1_9,  u2_9,  v1_9,  v2_9,  y1_9,  y2_9,  o_9,  gain_9,  bias_9  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain5, 0.0
        # Oscillator 10 (RHipRoll)
        u1_10, u2_10, v1_10, v2_10, y1_10, y2_10, o_10, gain_10, bias_10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain4, 0.0
        # Oscillator 11 (RlAnkleRoll)
        u1_11, u2_11, v1_11, v2_11, y1_11, y2_11, o_11, gain_11, bias_11 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain5, 0.0
        # Oscillator 12 (LShoulderPitch)
        u1_12, u2_12, v1_12, v2_12, y1_12, y2_12, o_12, gain_12, bias_12 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain6, bias4
        # Oscillator 13 (RShoulderPitch)
        u1_13, u2_13, v1_13, v2_13, y1_13, y2_13, o_13, gain_13, bias_13 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain6, bias4
    
        # create final output of each oscillator
        o1_list = list()
        o2_list = list()
        o3_list = list()
        o4_list = list()
        o5_list = list()
        o6_list = list()
        o7_list = list()
        o8_list = list()
        o9_list = list()
        o10_list = list()
        o11_list = list()
        o12_list = list()
        o13_list = list()
        # time list
        t_list = list()
        
        def oscillator_next(u1, u2, v1, v2, y1, y2, f1, f2, s1, s2, bias, gain, dt, name):
    
            """Calculates the state variables in the next time step"""
    
            d_u1_dt = (-u1 - w0 * y2 -beta*v1 + ue + f1 + a*s1)/tau
            d_v1_dt = (-v1 + y1)/tau_prime
            y1 = max([0.0, u1])
    
            d_u2_dt = (-u2 - w0*y1 -beta*v2 + ue + f2 + a*s2)/tau
            d_v2_dt = (-v2 + y2)/tau_prime
            y2 = max([0.0, u2])
    
            u1 += d_u1_dt * dt
            u2 += d_u2_dt * dt
            v1 += d_v1_dt * dt
            v2 += d_v2_dt * dt
            o_ = -m1 * y1 + m2 * y2
            print(name, ':', o_)
            o = bias + gain * o_
    
            return u1, u2, v1, v2, y1, y2, o

        #Enable the synchronous mode
        vrep.simxSynchronous(clientID,1)
        #Start Vrep Simulation
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

        # Get the initial postion 
        res,LHip_position = vrep.simxGetObjectPosition(clientID,lHipHandle,-1,vrep.simx_opmode_blocking)
        res,RHip_position = vrep.simxGetObjectPosition(clientID,rHipHandle,-1,vrep.simx_opmode_blocking)
        print("LHip_position:",LHip_position)
        print("RHip_position",RHip_position)
        #center_position=0.5*LHip_position+0.5*RHip_position
        #print("center_position:",center_position)
        
        #Get initial X,Y,Z of the Robot center (the midpoint of two hip pitch)
        start_x = 0.5*LHip_position[0]+0.5*RHip_position[0]
        start_y = 0.5*LHip_position[1]+0.5*RHip_position[1]
        start_z = 0.5*LHip_position[2]+0.5*RHip_position[2]
        print('center position x is',start_x) 
        print('center position y is',start_y) 
        print('center position z is',start_z) 


        for t in np.arange(0.0, 100.0,dt):
    
            # Increment the log time variable
            time_log = t
            start = time.time()
            print("simulation_t", t)
            print('start_t: ', start)
    
            # Caculate the current angels of the Right and Left HipPitch (as feedback)
            feedback_angles=[]
            res,LHip_angle = vrep.simxGetJointPosition(clientID,lHipHandle,vrep.simx_opmode_streaming)
            res,RHip_angle = vrep.simxGetJointPosition(clientID,rHipHandle,vrep.simx_opmode_streaming)
            feedback_angles.append(LHip_angle)
            feedback_angles.append(RHip_angle)
    
            # Calculate next state of oscillator 1 (pacemaker)
            f1_1, f2_1 = 0.0, 0.0
            s1_1, s2_1 = 0.0, 0.0
            u1_1, u2_1, v1_1, v2_1, y1_1, y2_1, o_1 = oscillator_next(u1=u1_1, u2=u2_1, v1=v1_1, v2=v2_1, y1=y1_1, y2=y2_1,
                                                                      f1=f1_1, f2=f2_1, s1=s1_1, s2=s2_1,
                                                                      bias=bias_1, gain=gain_1,
                                                                      dt=dt,name='pacemaker')
    
            # Calculate next state of oscillator 2
            # w_ij -> j=1 (oscillator 1) is master, i=2 (oscillator 2) is slave
            w_21 = 1.0
            f1_2, f2_2 = k * feedback_angles[0], -k * feedback_angles[0]
            s1_2, s2_2 = w_21*u1_1, w_21*u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_2, u2_2, v1_2, v2_2, y1_2, y2_2, o_2 = oscillator_next(u1=u1_2, u2=u2_2, v1=v1_2, v2=v2_2, y1=y1_2, y2=y2_2,
                                                                      f1=f1_2, f2=f2_2, s1=s1_2, s2=s2_2,
                                                                      bias=bias_2, gain=1.0*gain_2,
                                                                      dt=dt,name='Lhip')
            # Calculate next state of oscillator 3
            # w_ij -> j=1 (oscillator 1) is master, i=3 (oscillator 3) is slave
            w_31 = -1.0
            f1_3, f2_3 = k * feedback_angles[1], -k * feedback_angles[1]
            s1_3, s2_3 = w_31*u1_1, w_31*u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_3, u2_3, v1_3, v2_3, y1_3, y2_3, o_3 = oscillator_next(u1=u1_3, u2=u2_3, v1=v1_3, v2=v2_3, y1=y1_3, y2=y2_3,
                                                                      f1=f1_3, f2=f2_3, s1=s1_3, s2=s2_3,
                                                                      bias=bias_3, gain=gain_3,
                                                                      dt=dt,name='Rhip')
            # Calculate next state of oscillator 4
            # w_ij -> j=2 (oscillator 2) is master, i=4 (oscillator 4) is slave
            w_42 = -1.0
            f1_4, f2_4 = 0.0, 0.0
            s1_4, s2_4 = w_42*u1_2, w_42*u2_2  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_4, u2_4, v1_4, v2_4, y1_4, y2_4, o_4 = oscillator_next(u1=u1_4, u2=u2_4, v1=v1_4, v2=v2_4, y1=y1_4, y2=y2_4,
                                                                      f1=f1_4, f2=f2_4, s1=s1_4, s2=s2_4,
                                                                      bias=bias_4, gain=gain_4,
                                                                      dt=dt,name='Lknee')
    
            # Calculate next state of oscillator 5
            # w_ij -> j=3 (oscillator 3) is master, i=5 (oscillator 5) is slave
            w_53 = -1.0
            f1_5, f2_5 = 0.0, 0.0
            s1_5, s2_5 = w_53*u1_3, w_53*u2_3  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_5, u2_5, v1_5, v2_5, y1_5, y2_5, o_5 = oscillator_next(u1=u1_5, u2=u2_5, v1=v1_5, v2=v2_5, y1=y1_5, y2=y2_5,
                                                                      f1=f1_5, f2=f2_5, s1=s1_5, s2=s2_5,
                                                                      bias=bias_5, gain=gain_5,
                                                                      dt=dt,name='Rknee')
    
            # Calculate next state of oscillator 6
            # w_ij -> j=2 (oscillator 2) is master, i=6 (oscillator 6) is slave
            w_62 = -1.0
            f1_6, f2_6 = 0.0, 0.0
            s1_6, s2_6 = w_62*u1_2, w_62*u2_2  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_6, u2_6, v1_6, v2_6, y1_6, y2_6, o_6 = oscillator_next(u1=u1_6, u2=u2_6, v1=v1_6, v2=v2_6, y1=y1_6, y2=y2_6,
                                                                      f1=f1_6, f2=f2_6, s1=s1_6, s2=s2_6,
                                                                      bias=bias_6, gain=gain_6,
                                                                      dt=dt,name='Lankle')
    
            # Calculate next state of oscillator 7
            # w_ij -> j=3 (oscillator 3) is master, i=7 (oscillator 7) is slave
            w_73 = -1.0
            f1_7, f2_7 = 0.0, 0.0
            s1_7, s2_7 = w_73*u1_3, w_73*u2_3  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_7, u2_7, v1_7, v2_7, y1_7, y2_7, o_7 = oscillator_next(u1=u1_7, u2=u2_7, v1=v1_7, v2=v2_7, y1=y1_7, y2=y2_7,
                                                                      f1=f1_7, f2=f2_7, s1=s1_7, s2=s2_7,
                                                                      bias=bias_7, gain=gain_7,
                                                                      dt=dt,name='Rankle')
    
            # Calculate next state of oscillator 8
            # w_ij -> j=1 (oscillator 1) is master, i=8 (oscillator 8) is slave
            w_81 = 1.0
            f1_8, f2_8 = 0.0, 0.0
            s1_8, s2_8 = w_81*u1_1, w_81*u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_8, u2_8, v1_8, v2_8, y1_8, y2_8, o_8 = oscillator_next(u1=u1_8, u2=u2_8, v1=v1_8, v2=v2_8, y1=y1_8, y2=y2_8,
                                                                      f1=f1_8, f2=f2_8, s1=s1_8, s2=s2_8,
                                                                      bias=bias_8, gain=gain_8,
                                                                      dt=dt,name='LhipRoll')
    
            # Calculate next state of oscillator 9
            # w_ij -> j=8 (oscillator 8) is master, i=9 (oscillator 9) is slave
            w_98 = -1.0
            f1_9, f2_9 = 0.0, 0.0
            s1_9, s2_9 = w_98*u1_8, w_98*u2_8  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_9, u2_9, v1_9, v2_9, y1_9, y2_9, o_9 = oscillator_next(u1=u1_9, u2=u2_9, v1=v1_9, v2=v2_9, y1=y1_9, y2=y2_9,
                                                                      f1=f1_9, f2=f2_9, s1=s1_9, s2=s2_9,
                                                                      bias=bias_9, gain=gain_9,
                                                                      dt=dt,name='RhipRoll')
    
            # Calculate next state of oscillator 10
            # w_ij -> j=1 (oscillator 1) is master, i=10 (oscillator 10) is slave
            w_101 = 1.0
            f1_10, f2_10 = 0.0, 0.0
            s1_10, s2_10 = w_101*u1_1, w_101*u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_10, u2_10, v1_10, v2_10, y1_10, y2_10, o_10 = oscillator_next(u1=u1_10, u2=u2_10, v1=v1_10, v2=v2_10, y1=y1_10, y2=y2_10,
                                                                             f1=f1_10, f2=f2_10, s1=s1_10, s2=s2_10,
                                                                             bias=bias_10, gain=gain_10,
                                                                             dt=dt,name='LankleRoll')
    
            # Calculate next state of oscillator 11
            # w_ij -> j=10 (oscillator 10) is master, i=11 (oscillator 11) is slave
            w_1110 = -1.0
            f1_11, f2_11 = 0.0, 0.0
            s1_11, s2_11 = w_1110*u1_10, w_1110*u2_10  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_11, u2_11, v1_11, v2_11, y1_11, y2_11, o_11 = oscillator_next(u1=u1_11, u2=u2_11, v1=v1_11, v2=v2_11, y1=y1_11, y2=y2_11,
                                                                             f1=f1_11, f2=f2_11, s1=s1_11, s2=s2_11,
                                                                             bias=bias_11, gain=gain_11,
                                                                             dt=dt,name='RankleRoll')
    
            # Calculate next state of oscillator 12
            # w_ij -> j=1 (oscillator 1) is master, i=12 (oscillator 12) is slave
            w_121 = -1.0
            f1_12, f2_12 = 0.0, 0.0
            s1_12, s2_12 = w_121*u1_1, w_121*u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_12, u2_12, v1_12, v2_12, y1_12, y2_12, o_12 = oscillator_next(u1=u1_12, u2=u2_12, v1=v1_12, v2=v2_12, y1=y1_12, y2=y2_12,
                                                                             f1=f1_12, f2=f2_12, s1=s1_12, s2=s2_12,
                                                                             bias=bias_12, gain=gain_12,
                                                                             dt=dt,name='Lshoulder')
            #print('o_12:',o_12)
    
            # Calculate next state of oscillator 13
            # w_ij -> j=1 (oscillator 1) is master, i=13 (oscillator 13) is slave
            w_131 = 1.0
            f1_13, f2_13 = 0.0, 0.0
            s1_13, s2_13 = w_131*u1_1, w_131*u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            u1_13, u2_13, v1_13, v2_13, y1_13, y2_13, o_13 = oscillator_next(u1=u1_13, u2=u2_13, v1=v1_13, v2=v2_13, y1=y1_13, y2=y2_13,
                                                                             f1=f1_13, f2=f2_13, s1=s1_13, s2=s2_13,
                                                                             bias=bias_13, gain=gain_13,
                                                                             dt=dt,name='Rshoulder')
    
            
            # oi_list is used for the plot
            o1_list.append(o_1)
            o2_list.append(o_2)
            o3_list.append(o_3)
            o4_list.append(o_4)
            o5_list.append(o_5)
            o6_list.append(o_6)
            o7_list.append(o_7)
            o8_list.append(o_8)
            o9_list.append(o_9)
            o10_list.append(o_10)
            o11_list.append(o_11)
            o12_list.append(o_12)
            o13_list.append(o_13)
            t_list.append(t)

            #Get the velocity of the object

            res, LV, AV = vrep.simxGetObjectVelocity(clientID, NAOHandle, vrep.simx_opmode_streaming)
            #print("LV",LV)
            #Set joints positions from CPG
            current_angles = {
                'LHipPitch3': o_2,
                'RHipPitch3': o_3,
                'LKneePitch3': o_4,
                'RKneePitch3': o_5,
                'LAnklePitch3': o_6,
                'RAnklePitch3': o_7,
                'LHipRoll3': o_8,
                'LAnkleRoll3': o_9,
                'RHipRoll3': o_10,
                'RAnkleRoll3': o_11,
                'LShoulderPitch3': o_12,
                'RShoulderPitch3': o_13
            }
            # current_angles = {
            #     'LHipPitch3': o_2,
            #     'RHipPitch3': o_3,
            #     'LKneePitch3': o_4,
            #     'RKneePitch3': o_5,
            #     'LAnklePitch3': -o_2-o_4,
            #     'RAnklePitch3': -o_3-o_5,
            #     'LShoulderPitch3': o_12,
            #     'RShoulderPitch3': o_13
            # }
            print('--------------------------------------------')
            for item in current_angles:
                print(item, ":", current_angles[item] * 180 / 3.14)
        

            # Set Joints Angles robot_handle.set_angles(current_angles)

            #time.sleep(dt)
            handle_dict = {}
            #vrep.simxPauseCommunication(self.clientID,True)
            for joint_name in current_angles.keys():
                res,handle_dict[joint_name]= vrep.simxGetObjectHandle(clientID,joint_name,vrep.simx_opmode_blocking)
            #for Joint_name in current_angles.keys():
                #Get the Force of each joint:
                # res,force=vrep.simxGetJointForce(clientID,handle_dict[joint_name],vrep.simx_opmode_blocking)
                #force_list.append(force)
                res =vrep.simxSetJointTargetPosition(clientID,handle_dict[joint_name],current_angles[joint_name], vrep.simx_opmode_oneshot)
                # t1=time.time()
                # if joint_name == 'RShoulderPitch3': print('t1', t1 - start)
            # Check if the NAO has fallen, if so, end the for-loop
            height_threshold = 0.9 * start_z
            res,LHip_position = vrep.simxGetObjectPosition(clientID,lHipHandle,-1,vrep.simx_opmode_blocking)
            res,RHip_position = vrep.simxGetObjectPosition(clientID,rHipHandle,-1,vrep.simx_opmode_blocking)
            end_x = 0.5 * LHip_position[0] + 0.5 * RHip_position[0]
            end_y = 0.5 * LHip_position[1] + 0.5 * RHip_position[1]
            end_z = 0.5 * LHip_position[2] + 0.5 * RHip_position[2]
            #print("final position of the center of %s time is: (%s  %s  %s)" %(t,end_x,end_y,end_z))

            #Get the object orientation

            res,rot_y_temp =vrep.simxGetObjectOrientation(clientID,NAOHandle,-1,vrep.simx_opmode_streaming)

            # caculate the sum of the orientation
            """
            if rot_y_temp[1] <= 0:
                rot_y_sum1 += rot_y_temp[1]
            else:
                rot_y_sum2 += rot_y_temp[1]
            """
            # Check fallen
            if end_z < height_threshold:
                print("***********Fallen***********")
                break
            #print("target:",o_2)
            #sm_t_list.append(o_2)
            vrep.simxSynchronousTrigger(clientID)
            t2=time.time()
            print('t2',t2-start)


        # Displaying the walking time
        print("Accurate time is:",time_log)
    
        #Caculate the fitness
        # Extract the walking time for fitness
        uptime = time_log
        #rot_y =abs(rot_y_sum1)+rot_y_sum2
        #print("SUM OF ROT_Y is", rot_y)
        fitness=0.0
        if uptime ==0.0:
            fitness = 0.0
            print("no walking time, fitness is 0.0")
        else:
            fitness = calc_fitness(start_x=start_x, start_y=start_y, start_z=start_z,
                                   end_x=end_x, end_y=end_y, end_z=end_z,uptime=uptime
                                   )
            print("fitness is ", fitness)
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking) #blocking mode: wait for the actual reply
        vrep.simxFinish(clientID)
    else:
        print('Failed connecting to remote API server')
        print('Program ended')

    return fitness
