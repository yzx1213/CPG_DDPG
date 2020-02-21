# Sim-to-Real Model-Based Reinforcement Learning for Biped Locomotion
By: Ye Zhixing, Gao Tian

In this project, we use a more simplied and flexible model named CPG to control the bipedal robot NAO's locomotion. We implement the CPG network controller in the V-REP simulation as well as on the real world. Though the outcome of real NAO is not optimal, we nd the shortcomings of our work and we may have some improvement directions. Furthermore, we try to let NAO go more straightly by utilizing a reinforcement learning model DDPG, which can control the CPG network by tuning several parameters.

The project is based on **python**. To run the code, please download the simulation platform V-REP first and load "main_script.ttt", then run the main program "CPG_DDPG.py" to see the NAO's performance. 
