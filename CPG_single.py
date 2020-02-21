import numpy as np
import os
import csv
import time
import vrep 
import random
#from Fitness import calc_fitness
class oscillator_nw(object):
    def __init__(self, position_vector, clientID):
        if clientID != -1:
            self.clientID = clientID
            


            Ctrl_Joints = ['LHipPitch3','RHipPitch3','LKneePitch3','RKneePitch3','LAnklePitch3','RAnklePitch3','LHipRoll3',
                       'LAnkleRoll3','RHipRoll3','RAnkleRoll3','LShoulderPitch3','RShoulderPitch3']
        
            #handle the NAO object 
            res, NAOHandle = vrep.simxGetObjectHandle(self.clientID, "NAO", vrep.simx_opmode_blocking)
            
            # the initial x position of the NAO is -1.65

            #handle the CtrlJoints
            self.Handle = [None] * len(Ctrl_Joints)
            for i in range (len(Ctrl_Joints)):
                res, self.Handle[i] = vrep.simxGetObjectHandle(self.clientID,Ctrl_Joints[i],vrep.simx_opmode_blocking)

            # Handle the joints to obtain the position used to check the falling 
            res, self.lHipHandle = vrep.simxGetObjectHandle(self.clientID,'LHipPitch3',vrep.simx_opmode_blocking)
            res, self.rHipHandle = vrep.simxGetObjectHandle(self.clientID,'RHipPitch3',vrep.simx_opmode_blocking)
    	    
            self.tau = 0.3           # rise time constant, determine the frequency 
            self.tau_prime = 1   # adaptation effect time constant
            self.beta =  5           # self-inhibition constant
            self.w0 = 2.2829#2.2829          # constant sof mutation inhibition between flexor and extensor
            self.ue = 0.4111          # tonic input to neurons
            self.m1 = 2             # weights used to calculate output (extensor neuron)
            self.m2 = 1             # weights used to calculate  output (flexor neuron)
            self.a = 1.0              # constant of interaction between oscillators

            self.kf=position_vector[0]
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
            self.k=position_vector[11]   

            self.tau = self.tau* self.kf
            self.tau_prime = self.tau_prime * self.kf

            self.u1_1,  self.u2_1,  self.v1_1,  self.v2_1,  self.y1_1,  self.y2_1,  self.o_1,  self.gain_1,  self.bias_1  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
            # Oscillator 2 (LHipPitch, angle feedback)
            self.u1_2,  self.u2_2,  self.v1_2,  self.v2_2,  self.y1_2,  self.y2_2,  self.o_2,  self.gain_2,  self.bias_2  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain1, bias1
            # Oscillator 3 (RHipPitch,angle feedback)
            self.u1_3,  self.u2_3,  self.v1_3,  self.v2_3,  self.y1_3,  self.y2_3,  self.o_3,  self.gain_3,  self.bias_3  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain1, bias1
            # Oscillator 4 (LKneePitch)
            self.u1_4,  self.u2_4,  self.v1_4,  self.v2_4,  self.y1_4,  self.y2_4,  self.o_4,  self.gain_4,  self.bias_4  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain3, bias2
            # Oscillator 5 (RKneePitch)
            self.u1_5,  self.u2_5,  self.v1_5,  self.v2_5,  self.y1_5,  self.y2_5,  self.o_5,  self.gain_5,  self.bias_5  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain3, bias2
            # Oscillator 6 (LAnklePitch)
            self.u1_6,  self.u2_6,  self.v1_6,  self.v2_6,  self.y1_6,  self.y2_6,  self.o_6,  self.gain_6,  self.bias_6  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain2, bias3
            # Oscillator 7 (RAnklePitch)
            self.u1_7,  self.u2_7,  self.v1_7,  self.v2_7,  self.y1_7,  self.y2_7,  self.o_7,  self.gain_7,  self.bias_7  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain2, bias3
            # Oscillator 8 (LHipRoll)
            self.u1_8,  self.u2_8,  self.v1_8,  self.v2_8,  self.y1_8,  self.y2_8,  self.o_8,  self.gain_8,  self.bias_8  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain4, 0.0
            # Oscillator 9 (lAnkleRoll)
            self.u1_9,  self.u2_9,  self.v1_9,  self.v2_9,  self.y1_9,  self.y2_9,  self.o_9,  self.gain_9,  self.bias_9  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain5, 0.0
            # Oscillator 10 (RHipRoll)
            self.u1_10, self.u2_10, self.v1_10, self.v2_10, self.y1_10, self.y2_10, self.o_10, self.gain_10, self.bias_10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain4, 0.0
            # Oscillator 11 (RlAnkleRoll)
            self.u1_11, self.u2_11, self.v1_11, self.v2_11, self.y1_11, self.y2_11, self.o_11, self.gain_11, self.bias_11 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain5, 0.0
            # Oscillator 12 (LShoulderPitch)
            self.u1_12, self.u2_12, self.v1_12, self.v2_12, self.y1_12, self.y2_12, self.o_12, self.gain_12, self.bias_12 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain6, bias4
            # Oscillator 13 (RShoulderPitch)
            self.u1_13, self.u2_13, self.v1_13, self.v2_13, self.y1_13, self.y2_13, self.o_13, self.gain_13, self.bias_13 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gain6, bias4
    
    def oscillator_next(self, u1, u2, v1, v2, y1, y2, f1, f2, s1, s2, bias, gain, dt):
    
            """Calculates the state variables in the next time step"""
    
            d_u1_dt = (-u1 - self.w0 * y2 -self.beta*v1 + self.ue + f1 + self.a*s1)/self.tau # a???
            d_v1_dt = (-v1 + y1)/self.tau_prime
            y1 = max([0.0, u1])
    
            d_u2_dt = (-u2 - self.w0*y1 -self.beta*v2 + self.ue + f2 + self.a*s2)/self.tau
            d_v2_dt = (-v2 + y2)/self.tau_prime
            y2 = max([0.0, u2])
    
            u1 += d_u1_dt * dt
            u2 += d_u2_dt * dt
            v1 += d_v1_dt * dt
            v2 += d_v2_dt * dt
    
            o = bias + gain*(-self.m1*y1 + self.m2*y2)
    
            return u1, u2, v1, v2, y1, y2, o


        #define ocilator (Ref:Cristiano 2014 Locomotion Control of a Biped Robot Through a Feedback CPG Network )
    def oscillator_step(self, para_vector):
        

        """Caculate the CPG model and return the fitness for Genetic Algorithm"""
        #Constants in Matsuoka Ocilator Model
        self.kf=para_vector[0]
        gain1=para_vector[1]
        gain2=para_vector[2]
        gain3=para_vector[3]
        gain4=para_vector[4]
        gain5=para_vector[5]
        gain6=para_vector[6]
        bias1=para_vector[7]
        bias2=para_vector[8]
        bias3=para_vector[9]
        bias4=para_vector[10]
        self.k=para_vector[11]   

        # self.tau = self.tau* self.kf
        # self.tau_prime = self.tau_prime * self.kf

        self.gain_2, self.bias_2 = gain1, bias1
        self.gain_3, self.bias_3 = gain1, bias1
        self.gain_4, self.bias_4 = gain3, bias2
        self.gain_5, self.bias_5 = gain3, bias2
        self.gain_6, self.bias_6 = gain2, bias3
        self.gain_7, self.bias_7 = gain2, bias3
        self.gain_8, self.bias_8 = gain4, 0
        self.gain_9, self.bias_9 = gain5, 0        
        self.gain_10, self.bias_10 = gain4, 0
        self.gain_11, self.bias_11 = gain5, 0
        self.gain_12, self.bias_12 = gain6, bias4
        self.gain_13, self.bias_13 = gain6, bias4
        # print(self.gain_2,self.gain_3,self.gain_4,self.gain_5,self.gain_6,self.gain_7,self.gain_8,self.gain_9,self.gain_10,self.gain_11,self.gain_12,self.gain_13)
        

        # kf is a constant to modulate the time constant
        
    
        # Define step time
        dt = 0.01   #(50ms)

        # Variables in Matsuoka Ocilator Model (subscript 1 for extensor, 2 for flexor)
    
        
        # Caculate the current angels of the Right and Left HipPitch (as feedback)
        feedback_angles=[]
        res,LHip_angle = vrep.simxGetJointPosition(self.clientID,self.lHipHandle,vrep.simx_opmode_streaming)
        res, RHip_angle = vrep.simxGetJointPosition(self.clientID, self.rHipHandle, vrep.simx_opmode_streaming)
        # print('res':res)
        feedback_angles.append(LHip_angle)
        feedback_angles.append(RHip_angle)

        # Calculate next state of oscillator 1 (pacemaker)
        f1_1, f2_1 = 0.0, 0.0
        s1_1, s2_1 = 0.0, 0.0
        self.u1_1, self.u2_1, self.v1_1, self.v2_1, self.y1_1, self.y2_1, self.o_1 = self.oscillator_next(u1=self.u1_1, u2=self.u2_1, v1=self.v1_1, v2=self.v2_1, y1=self.y1_1, y2=self.y2_1,
                                                                    f1=f1_1, f2=f2_1, s1=s1_1, s2=s2_1,
                                                                    bias=self.bias_1, gain=self.gain_1,
                                                                    dt=dt)
        # print(self.u1_1, self.u2_1, self.v1_1, self.v2_1)

        # Calculate next state of oscillator 2
        # w_ij -> j=1 (oscillator 1) is master, i=2 (oscillator 2) is slave
        w_21 = 1.0
        f1_2, f2_2 = self.k * feedback_angles[0], -self.k * feedback_angles[0]
        s1_2, s2_2 = w_21*self.u1_1, w_21*self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_2, self.u2_2, self.v1_2, self.v2_2, self.y1_2, self.y2_2, self.o_2 = self.oscillator_next(u1=self.u1_2, u2=self.u2_2, v1=self.v1_2, v2=self.v2_2, y1=self.y1_2, y2=self.y2_2,
                                                                    f1=f1_2, f2=f2_2, s1=s1_2, s2=s2_2,
                                                                    bias=self.bias_2, gain=self.gain_2,
                                                                    dt=dt)

        # Calculate next state of oscillator 3
        # w_ij -> j=1 (oscillator 1) is master, i=3 (oscillator 3) is slave
        w_31 = -1.0
        f1_3, f2_3 = self.k * feedback_angles[1], -self.k * feedback_angles[1]
        s1_3, s2_3 = w_31*self.u1_1, w_31*self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_3, self.u2_3, self.v1_3, self.v2_3, self.y1_3, self.y2_3, self.o_3 = self.oscillator_next(u1=self.u1_3, u2=self.u2_3, v1=self.v1_3, v2=self.v2_3, y1=self.y1_3, y2=self.y2_3,
                                                                    f1=f1_3, f2=f2_3, s1=s1_3, s2=s2_3,
                                                                    bias=self.bias_3, gain=self.gain_3,
                                                                    dt=dt)

        # Calculate next state of oscillator 4
        # w_ij -> j=2 (oscillator 2) is master, i=4 (oscillator 4) is slave
        w_42 = -1.0
        f1_4, f2_4 = 0.0, 0.0
        s1_4, s2_4 = w_42*self.u1_2, w_42*self.u2_2  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_4, self.u2_4, self.v1_4, self.v2_4, self.y1_4, self.y2_4, self.o_4 = self.oscillator_next(u1=self.u1_4, u2=self.u2_4, v1=self.v1_4, v2=self.v2_4, y1=self.y1_4, y2=self.y2_4,
                                                                    f1=f1_4, f2=f2_4, s1=s1_4, s2=s2_4,
                                                                    bias=self.bias_4, gain=self.gain_4,
                                                                    dt=dt)

        # Calculate next state of oscillator 5
        # w_ij -> j=3 (oscillator 3) is master, i=5 (oscillator 5) is slave
        w_53 = -1.0
        f1_5, f2_5 = 0.0, 0.0
        s1_5, s2_5 = w_53*self.u1_3, w_53*self.u2_3  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_5, self.u2_5, self.v1_5, self.v2_5, self.y1_5, self.y2_5, self.o_5 = self.oscillator_next(u1=self.u1_5, u2=self.u2_5, v1=self.v1_5, v2=self.v2_5, y1=self.y1_5, y2=self.y2_5,
                                                                    f1=f1_5, f2=f2_5, s1=s1_5, s2=s2_5,
                                                                    bias=self.bias_5, gain=self.gain_5,
                                                                    dt=dt)

        # Calculate next state of oscillator 6
        # w_ij -> j=2 (oscillator 2) is master, i=6 (oscillator 6) is slave
        w_62 = -1.0
        f1_6, f2_6 = 0.0, 0.0
        s1_6, s2_6 = w_62*self.u1_2, w_62*self.u2_2  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_6, self.u2_6, self.v1_6, self.v2_6, self.y1_6, self.y2_6, self.o_6 = self.oscillator_next(u1=self.u1_6, u2=self.u2_6, v1=self.v1_6, v2=self.v2_6, y1=self.y1_6, y2=self.y2_6,
                                                                    f1=f1_6, f2=f2_6, s1=s1_6, s2=s2_6,
                                                                    bias=self.bias_6, gain=self.gain_6,
                                                                    dt=dt)

        # Calculate next state of oscillator 7
        # w_ij -> j=3 (oscillator 3) is master, i=7 (oscillator 7) is slave
        w_73 = -1.0
        f1_7, f2_7 = 0.0, 0.0
        s1_7, s2_7 = w_73*self.u1_3, w_73*self.u2_3  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_7, self.u2_7, self.v1_7, self.v2_7, self.y1_7, self.y2_7, self.o_7 = self.oscillator_next(u1=self.u1_7, u2=self.u2_7, v1=self.v1_7, v2=self.v2_7, y1=self.y1_7, y2=self.y2_7,
                                                                    f1=f1_7, f2=f2_7, s1=s1_7, s2=s2_7,
                                                                    bias=self.bias_7, gain=self.gain_7,
                                                                    dt=dt)

        # Calculate next state of oscillator 8
        # w_ij -> j=1 (oscillator 1) is master, i=8 (oscillator 8) is slave
        w_81 = 1.0
        f1_8, f2_8 = 0.0, 0.0
        s1_8, s2_8 = w_81*self.u1_1, w_81*self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_8, self.u2_8, self.v1_8, self.v2_8, self.y1_8, self.y2_8, self.o_8 = self.oscillator_next(u1=self.u1_8, u2=self.u2_8, v1=self.v1_8, v2=self.v2_8, y1=self.y1_8, y2=self.y2_8,
                                                                    f1=f1_8, f2=f2_8, s1=s1_8, s2=s2_8,
                                                                    bias=self.bias_8, gain=self.gain_8,
                                                                    dt=dt)

        # Calculate next state of oscillator 9
        # w_ij -> j=8 (oscillator 8) is master, i=9 (oscillator 9) is slave
        w_98 = -1.0
        f1_9, f2_9 = 0.0, 0.0
        s1_9, s2_9 = w_98*self.u1_8, w_98*self.u2_8  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_9, self.u2_9, self.v1_9, self.v2_9, self.y1_9, self.y2_9, self.o_9 = self.oscillator_next(u1=self.u1_9, u2=self.u2_9, v1=self.v1_9, v2=self.v2_9, y1=self.y1_9, y2=self.y2_9,
                                                                    f1=f1_9, f2=f2_9, s1=s1_9, s2=s2_9,
                                                                    bias=self.bias_9, gain=self.gain_9,
                                                                    dt=dt)

        # Calculate next state of oscillator 10
        # w_ij -> j=1 (oscillator 1) is master, i=10 (oscillator 10) is slave
        w_101 = 1.0
        f1_10, f2_10 = 0.0, 0.0
        s1_10, s2_10 = w_101*self.u1_1, w_101*self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_10, self.u2_10, self.v1_10, self.v2_10, self.y1_10, self.y2_10, self.o_10 = self.oscillator_next(u1=self.u1_10, u2=self.u2_10, v1=self.v1_10, v2=self.v2_10, y1=self.y1_10, y2=self.y2_10,
                                                                            f1=f1_10, f2=f2_10, s1=s1_10, s2=s2_10,
                                                                            bias=self.bias_10, gain=self.gain_10,
                                                                            dt=dt)

        # Calculate next state of oscillator 11
        # w_ij -> j=10 (oscillator 10) is master, i=11 (oscillator 11) is slave
        w_1110 = -1.0
        f1_11, f2_11 = 0.0, 0.0
        s1_11, s2_11 = w_1110*self.u1_10, w_1110*self.u2_10  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_11, self.u2_11, self.v1_11, self.v2_11, self.y1_11, self.y2_11, self.o_11 = self.oscillator_next(u1=self.u1_11, u2=self.u2_11, v1=self.v1_11, v2=self.v2_11, y1=self.y1_11, y2=self.y2_11,
                                                                            f1=f1_11, f2=f2_11, s1=s1_11, s2=s2_11,
                                                                            bias=self.bias_11, gain=self.gain_11,
                                                                            dt=dt)

        # Calculate next state of oscillator 12
        # w_ij -> j=1 (oscillator 1) is master, i=12 (oscillator 12) is slave
        w_121 = -1.0
        f1_12, f2_12 = 0.0, 0.0
        s1_12, s2_12 = w_121*self.u1_1, w_121*self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_12, self.u2_12, self.v1_12, self.v2_12, self.y1_12, self.y2_12, self.o_12 = self.oscillator_next(u1=self.u1_12, u2=self.u2_12, v1=self.v1_12, v2=self.v2_12, y1=self.y1_12, y2=self.y2_12,
                                                                            f1=f1_12, f2=f2_12, s1=s1_12, s2=s2_12,
                                                                            bias=self.bias_12, gain=self.gain_12,
                                                                            dt=dt)

        # Calculate next state of oscillator 13
        # w_ij -> j=1 (oscillator 1) is master, i=13 (oscillator 13) is slave
        w_131 = 1.0
        f1_13, f2_13 = 0.0, 0.0
        s1_13, s2_13 = w_131*self.u1_1, w_131*self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        self.u1_13, self.u2_13, self.v1_13, self.v2_13, self.y1_13, self.y2_13, self.o_13 = self.oscillator_next(u1=self.u1_13, u2=self.u2_13, v1=self.v1_13, v2=self.v2_13, y1=self.y1_13, y2=self.y2_13,
                                                                            f1=f1_13, f2=f2_13, s1=s1_13, s2=s2_13,
                                                                            bias=self.bias_13, gain=self.gain_13,
                                                                            dt=dt)

        #Set joints positions from CPG
        current_angles = {
            'LHipPitch3': self.o_2,
            'RHipPitch3': self.o_3,
            'LKneePitch3': self.o_4,
            'RKneePitch3': self.o_5,
            'LAnklePitch3': self.o_6,
            'RAnklePitch3': self.o_7,
            'LHipRoll3': self.o_8,
            'LAnkleRoll3': self.o_9,
            'RHipRoll3': self.o_10,
            'RAnkleRoll3': self.o_11,
            'LShoulderPitch3': self.o_12,
            'RShoulderPitch3': self.o_13
        }
        # print(current_angles)


        # get all the handle number in a dict
        #force_list=[]
        handle_dict = {}
        #vrep.simxPauseCommunication(self.clientID,True)
        for joint_name in current_angles.keys():
            res,handle_dict[joint_name]= vrep.simxGetObjectHandle(self.clientID,joint_name,vrep.simx_opmode_blocking)
        #for Joint_name in current_angles.keys():
            #Get the Force of each joint:
            # res,force=vrep.simxGetJointForce(self.clientID,handle_dict[joint_name],vrep.simx_opmode_blocking)
            #force_list.append(force)
            res =vrep.simxSetJointTargetPosition(self.clientID,handle_dict[joint_name],current_angles[joint_name], vrep.simx_opmode_oneshot)
        return current_angles['LHipPitch3']