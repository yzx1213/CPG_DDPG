import numpy as np

def calc_fitness(start_x, start_y, start_z, end_x, end_y, end_z,uptime):
    print("start_x, start_y, start_z:",start_x, start_y, start_z)
    print(" end_x, end_y, end_z:", end_x, end_y, end_z)
    print("uptime is :", uptime)
    x_distance = end_x -start_x
    y_distance = end_y -start_y

    x_vel = x_distance /uptime

    fitness = 0.0
    fitness = x_distance + uptime * 0.3   # 0.5 is the constant for beter results

    return fitness
