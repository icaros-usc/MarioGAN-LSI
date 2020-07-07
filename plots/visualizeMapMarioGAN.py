#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
import toml
import argparse
import os
import sys
from matplotlib.colors import ListedColormap
import seaborn as sns
from util.SearchHelper import *


gridsize_x = 151
gridsize_y = 26

from util.gan_generator import *

sys.path.append(os.getcwd())
import pathlib
os.environ['CLASSPATH']=os.path.join(str(pathlib.Path().absolute()),"Mario.jar")

from jnius import autoclass
MarioGame = autoclass('engine.core.MarioGame')
Agent = autoclass('agents.robinBaumgarten.Agent')
GAN_model_path = "GANTrain/samples/netG_epoch_4999_7684.pth"

num_params = 32
boundary_value = 1
batch_size = 1
nz = 32

def write_level(level, level_name):
    realLevel=to_level(level)

    text_file = open(level_name, "w")
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
    
    

#file_name = "CMAES_MarioGANBC_sim1_elites_freq20.csv"

#feature1Range = (0, 0.316)
#feature2Range = (0.212, 0.46)

#def compute_summary(fitnesses, num_cells, max_fitness):
#    QD_score = sum(fitnesses)
#    coverage = float(num_cells/ (gridsize * gridsize))
#    return QD_score, coverage, max_fitness
map = np.zeros((gridsize_x,gridsize_y))
mapIndx = np.zeros((gridsize_x,gridsize_y))




def write_level(level,level_name):
    realLevel=to_level(level)

    text_file = open(level_name, "w")
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
    


def import_data_file(file_name,feature_ranges):
    fitnesses = []
    num_cells = 0
    max_fitness = -1
    num_cells_max_fitness = 0


    with open(file_name) as csvfile:
            all_records = csv.reader(csvfile, delimiter=',')
            print("importing data from file: " + str(file_name))
            for i,one_map in enumerate(all_records):
                if i == 499:

                    c = np.zeros((gridsize_x,gridsize_y))

                    for data_point in one_map: 
                        #i=i+1
                        #from IPython import embed
                        #embed()
                        data_point=data_point[1:-1]
                        feature1Range = feature_ranges[0]
                        feature2Range = feature_ranges[1]
                        data_point_info=data_point.split(', ')
                        #if len(data_point_info) < 4:
                        #    sys.exit("Error: " + str("corrupted file"))

                        bc_1 = float(data_point_info[2])
                        bc_2 = float(data_point_info[3])
                        cell_x =(bc_1 - feature1Range[0])/(feature1Range[1]-feature1Range[0])
                        cell_y = (bc_2 - feature2Range[0])/(feature2Range[1]-feature2Range[0])
                        
                        #from IPython import embed
                        #embed()

                        fitness = float(data_point_info[1][1:-1])
#                        if fitness > max_fitness: 
#                            max_fitness = fitness

                        if fitness == 1.0:
                            num_cells_max_fitness = num_cells_max_fitness + 1

                        #cell_x = gridsize-1 - int(cell_x*gridsize)
                        cell_x = int(cell_x*gridsize_x)
                        cell_y = int(cell_y*gridsize_y)
                        if cell_x >= gridsize_x or cell_y >= gridsize_y: 
                          continue
                        map[cell_x, cell_y] = fitness
                        if fitness == 1.0:
                            mapIndx[cell_x,cell_y] =data_point_info[0] 

                        #cell_indx = int(data_point_info[0])
                        fitnesses.append(fitness)
                        num_cells = num_cells + 1

                # if i % 1000 == 0:
                #     #from IPython import embed
                #     #embed()
                #     print "i: " + str(i)
                #     print "QD score: " + str(sum(fitnesses))
                #     QD_scores.append(sum(fitnesses))
                #     coverages.append(float(num_cells)/ (gridsize * gridsize))
                #     print "coverage: " + str(float(num_cells)/ (gridsize * gridsize))
            #if map == []:
            #    sys.exit("Error:corrupted file!")
            QD_score = sum(fitnesses)
            coverage = float(num_cells/ (gridsize_x * gridsize_y))
            max_fitness_coverage = float(num_cells_max_fitness/ num_cells)

            print("QD score: " + str(QD_score))
            print("coverage: " + str(coverage))
            print("Maximum fitness coverage: " + str(max_fitness_coverage))
            return map, QD_score, coverage, max_fitness_coverage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    

   # rc('text', usetex=True)


    parser.add_argument('-c','--config', help='path of BC config file',required=True)
    opt = parser.parse_args()

    parser = argparse.ArgumentParser()
    elite_map_config=toml.load(opt.config)
    feature_ranges = []
    column_names = []
    bc_names = []
    for bc in elite_map_config["Map"]["Features"]:
      feature_ranges.append((bc["low"],bc["high"]))
      column_names.append(bc["name"])
      bc_names.append(bc["name"])
    
                    #if i == 999 and cell_x == 8 and cell_y == 21:
                    #    from IPython import embed
                    #    embed()
                    # if i == 999 and (cell_indx == 276 or cell_indx == 552 or cell_indx == 196 or cell_indx == 128):

    data_root = 'test_file/test_map_mariogan'
    files = sorted([f for f in os.listdir(data_root)])[0:]
    QD_scores = []
    coverages = []
    max_fitnesses_coverage = []
    for file_name in files:
    #file_name = files[1]
    #from IPython import embed
    #embed()
      file_name = str(data_root + "/" + file_name)
      map, QD_score, coverage, max_fitness_coverage = import_data_file(file_name,feature_ranges)
      QD_scores.append(QD_score)
      coverages.append(coverage)
      max_fitnesses_coverage.append(max_fitness_coverage)

    #from IPython import embed
    #embed()
#      cmap = sns.cm.rocket_r

      #map = mapT
      #cmap =
      np.max(np.nonzero(mapIndx[0,:])) 
      mapT = np.zeros([gridsize_y, gridsize_x])
      mapTindx = np.zeros([gridsize_y, gridsize_x])

      for i in range(0,gridsize_x):
        for j in range(0,gridsize_y):
            mapT[gridsize_y-1-j,i] = map[i,j]
            mapTindx[gridsize_y-1-j,i] = mapIndx[i,j]
     





      map = mapT


      cmap = ListedColormap(sns.color_palette("coolwarm", 5).as_hex())
      with sns.axes_style("white"):
        numTicks = 10#11
        plt.figure(figsize=(10,9))
        sns.set(font_scale=2.5)
        sns.set_style({'font.family':'serif', 'font.serif':'Palatino'})
        #map = np.flip(map,0)
        mask = np.zeros_like(map)
        zero_indices = np.where(map == 0)
        mask[zero_indices] = np.nan
        cmap.set_bad("white") 
        #plt.text(150.5, 1.08, "sdfsd",
        # horizontalalignment='center',
        # fontsize=20)    
        g = sns.heatmap(map, annot=False, fmt=".0f",
                #yticklabels=[],
                vmin=0.0,
                vmax=1,
                mask = mask,
                cmap = cmap,
                rasterized=True,
                #square = True,
                cbar = False)
                #vmin=np.nanmin(fitnessMap),
                #vmax=np.nanmax(fitnessMap))
        fig = g.get_figure()
        #g.set(xticks = [0,25])
        #g.set(xticklabels = ['0','25'])
        #g.set(yticks = [0,150])
        #g.set(yticklabels = [150,0])
    #plt.title("sfsdF",y = 0.1)
        #for ax, title in zip(g.axes.flat, col_order):
        #    ax.set_title(title)
        #    ax.text(0.85, 0.85,'Text Here', fontsize=9) #add text
        
        g.set(xlabel = "Number of Sky Tiles")
        g.set(ylabel = "Number of Enemies")
        #g.set(title = "CMA-ME")

        g.set(xticks = [0,150])
        g.set(xticklabels = ['0','150'])
        g.set(yticks = [0,25])
        g.set(yticklabels = [25,0])
#        plt.xticklabels=['0','4.17'], 
        for item in g.get_xticklabels():
            item.set_rotation(0)
        for item in g.get_yticklabels():
            item.set_rotation(0)

        fig.savefig('MarioGAN_map.pdf')

#                plt.xticks = [0,60],


      
        indx_1 = [24,np.max(np.nonzero(mapTindx[24,:]))]
        indx_2 = [np.min(np.nonzero(mapTindx[:,0])),0]
        indx_3 = [16,np.max(np.nonzero(mapTindx[16,:]))]
        lvl1_id = int(mapTindx[indx_1[0],indx_1[1]])
        lvl2_id = int(mapTindx[indx_2[0],indx_2[1]])
        lvl3_id = int(mapTindx[indx_3[0],indx_3[1]])

        new_map = np.zeros_like(map)
        new_map[indx_1[0],indx_1[1]] = 1.0
        new_map[indx_2[0],indx_2[1]] = 1.0
        new_map[indx_3[0],indx_3[1]] = 1.0

        new_mask = np.zeros_like(new_map)
        new_zero_indices = np.where(new_map == 0)
        new_mask[new_zero_indices] = np.nan
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        new_cmap = ListedColormap(sns.color_palette(flatui, 5).as_hex())
        new_cmap.set_bad("white") 
        new_g = sns.heatmap(new_map, annot=False, fmt=".0f",
                #yticklabels=[],
                vmin=0.0,
                vmax=1,
                mask = new_mask,
                cmap = new_cmap,
                rasterized=True,
                #square = True,
                cbar = False)
                #vmin=np.nanmin(fitnessMap),
                #vmax=np.nanmax(fitnessMap))
        new_fig = new_g.get_figure()
        #g.set(xticks = [0,25])
        #g.set(xticklabels = ['0','25'])
        #g.set(yticks = [0,150])
        #g.set(yticklabels = [150,0])
        
        new_g.set(xticks = [0,150])
        new_g.set(xticklabels = ['0','150'])
        new_g.set(yticks = [0,25])
        new_g.set(yticklabels = [25,0])


        new_fig.savefig('selected.pdf')

        matplotlib.rcParams.update({'font.size': 20})
        plt.show()


    data_root = 'test_file/test_sim_mariogan'
    files = sorted([f for f in os.listdir(data_root)])[0:]

    for file_name in files:
      file_name = str(data_root + "/" + file_name)

    #file_name = "CMAME_MarioGANBC_sim3_elites_freq20.csv"
    with open(file_name) as sim_csvfile:
        all_simulations = csv.reader(sim_csvfile, delimiter=',')
        for i, individual in enumerate(all_simulations):
            if individual[0] == str(lvl1_id):
                latent_vector= individual[2]
                print("playing id: " + str(individual[1]))
                print("sky tiles: " + str(individual[9]))
                print("enemies: " + str(individual[10]))

                new_vector = latent_vector[1:-1].split(' ') 
                filtered_vector = filter(lambda x: x!="", new_vector)
                list_vector = list(filtered_vector)
                float_list = [float(indx) for indx in list_vector]
                #from IPython import embed
                #embed()
                level=gan_generate(float_list,batch_size,nz,GAN_model_path)
                #print(str(level))
                write_level(level,"lvl1.txt")
                eval_mario(level)
            if individual[0] == str(lvl3_id):
                latent_vector= individual[2]
                print("playing id: " + str(individual[0]))
                print("sky tiles: " + str(individual[9]))
                print("enemies: " + str(individual[10]))
                new_vector = latent_vector[1:-1].split(' ') 
                filtered_vector = filter(lambda x: x!="", new_vector)
                list_vector = list(filtered_vector)
                float_list = [float(indx) for indx in list_vector]
                #from IPython import embed
                #embed()
                level=gan_generate(float_list,batch_size,nz,GAN_model_path)
                #print(str(level))
                write_level(level,"lvl3.txt")
                eval_mario(level)


    #from IPython import embed
    #embed()
    #plt.figure()
    #sns.heatmap(map,yticklabels = range(0,gridsize)[::-1], annot=False, fmt=".0f", vmin=0,vmax=1)
    #plt.show()

    #for i in range(0,gridsize_y):
    #    for j in range(0, gridsize_x):

    QD_scores_avg = np.average(QD_scores)
    coverages_avg = np.average(coverages)
    max_fitnesses_coverage_avg = np.average(max_fitnesses_coverage)
    print("Average QDscore: " + str(QD_scores_avg))
    print("Average Coverage: " + str(coverages_avg))
    print("Average Max fitness coverage: " + str(max_fitnesses_coverage_avg))

#    map, QD_scores_random, coverages_random = import_data_file(RANDOM_file_name,feature_ranges)               
#    map, QD_scores_cmame_1, coverages_cmame_1 = import_data_file(CMAME_file_name_1,feature_ranges)               
#    map, QD_scores_cmame_2, coverages_cmame_2 = import_data_file(CMAME_file_name_2,feature_ranges)               

