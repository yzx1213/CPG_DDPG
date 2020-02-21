# Sim-to-Real Model-Based Reinforcement Learning for Biped Locomotion
By: Ye Zhixing, Gao Tian

In this project, we use a more simplified and flexible model named CPG to control the bipedal robot NAO's locomotion. We implement the CPG network controller in the V-REP simulation as well as on the real world. Though the outcome of real NAO is not optimal, we find the shortcomings of our work and we may have some improvement directions. Furthermore, we try to let NAO go more straightly by utilizing a reinforcement learning model DDPG, which can control the CPG network by tuning several parameters.

The project is based on **python**. 
* To run the model based singly on CPG, please download the simulation platform V-REP first and load "main_script.ttt", then run "SA.py"
* To run the code including CPG and DDPG, open "main_script.ttt" in V-REP and run the main program "CPG_DDPG.py" to see the NAO's performance. 
