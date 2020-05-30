#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
import toml
import argparse
import os
import sys
from util.gan_generator import *
from util.SearchHelper import *

sys.path.append(os.getcwd())
import pathlib
os.environ['CLASSPATH']=os.path.join(str(pathlib.Path().absolute()),"Mario.jar")

from jnius import autoclass
MarioGame = autoclass('engine.core.MarioGame')
Agent = autoclass('agents.robinBaumgarten.Agent')

gridsize_x = 151
gridsize_y = 26

num_params = 96
boundary_value = 1
batch_size = 3
nz = 32
record_frequency=20

max_BC = -1000
max_BC_indx = -1000

GAN_model_path = "GANTrain/samples/netG_epoch_4999_7684.pth"



def write_level(level):
    realLevel=to_level(level)

    text_file = open("generated.txt", "w")
    text_file.write(realLevel)
    text_file.close()

def eval_mario(level):
    realLevel=to_level(level)

    JString = autoclass('java.lang.String')
    agent = Agent()
    game = MarioGame()


    #from IPython import embed
    #embed()
    result = game.runGame(agent, JString(realLevel), 20, 0, True)
    #print(result)
    messageReceived=str(result.getCompletionPercentage())+","
    messageReceived+=str(result.getNumJumps())+","
    messageReceived+=str(result.getKillsTotal())+","
    messageReceived+=str(result.getCurrentLives())+","
    messageReceived+=str(result.getNumCollectedTileCoins())+","
    messageReceived+=str(result.getRemainingTime())
    print(messageReceived)
    
    

def import_data_file(file_name, all_simulations_name):
    global max_BC, max_BC_indx
    fitnesses = []
    num_cells = 0
    max_fitness = -1
    num_cells_max_fitness = 0


    with open(file_name) as csvfile:
            all_records = csv.reader(csvfile, delimiter=',')
            with open(all_simulations_name) as sim_csvfile:
                all_simulations = csv.reader(sim_csvfile, delimiter=',')
                print("importing data from file: " + str(file_name))
                for i,one_map in enumerate(all_records):
                    if i == 499:
                        map = np.zeros((151,26))

                        for data_point in one_map: 
                            #i=i+1

                            data_point=data_point[1:-1]
                            #feature1Range = feature_ranges[0]
                            #feature2Range = feature_ranges[1]
                            data_point_info=data_point.split(', ')
                            #if len(data_point_info) < 4:
                            #    sys.exit("Error: " + str("corrupted file"))

                            bc_1 = float(data_point_info[2])
                            bc_2 = float(data_point_info[3])
                            #cell_x =(bc_1 - feature1Range[0])/(feature1Range[1]-feature1Range[0])
                            #cell_y = (bc_2 - feature2Range[0])/(feature2Range[1]-feature2Range[0])
                            fitness = float(data_point_info[1][1:-1])

                            #if bc_1 > max_BC and fitness == 1.0:               
                            #  max_BC = bc_1 
                            if bc_1 / 150.0 + bc_2 / 26.0 > max_BC and fitness == 1.0:               
                              max_BC = + bc_1 / 150.0 + bc_2 / 26.0
                              max_BC_indx = data_point_info[0]
                              print("datapointinfo: " + str(data_point_info))
                              print("BC 1: " + str(bc_1))
                              print("BC 2: " + str(bc_2))
                              print("max_BC: " + str(max_BC))
                for i, individual in enumerate(all_simulations):
                    if individual[0] == str(max_BC_indx):
                        #from IPython import embed
                        #embed()
                        latent_vector= individual[2]
                        new_vector = latent_vector[1:-1].split(' ') 
                        filtered_vector = filter(lambda x: x!="", new_vector)
                        list_vector = list(filtered_vector)
                        float_list = [float(indx) for indx in list_vector]
                        #from IPython import embed
                        #embed()
                        level=gan_generate(float_list,batch_size,nz,GAN_model_path)
                        #print(str(level))
                        write_level(level)
                        eval_mario(level)

       
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    domain = "MarioGAN"    


    data_root = 'test_logs/MarioGAN/CMAME'


    data_root = str("test_logs/" + domain + "/CMAME")

    file_name = "CMAME_" + str(domain) + "BC_sim3_elites_freq20.csv"

    #file_name = "CMAME_MarioGANBC_sim3_elites_freq20.csv"
    file_name = str(data_root + "/" + file_name)
    all_simulations_name = "CMAME_" + str(domain) + "BC_sim3_all_simulations.csv"
    all_simulations_name = str(data_root + "/" + all_simulations_name)

    import_data_file(file_name, all_simulations_name)



    print("max BC: "+str(max_BC))
    print("max BC index: "+str(max_BC_indx))



