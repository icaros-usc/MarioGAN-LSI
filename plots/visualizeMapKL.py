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
gridsize = 60

#file_name = "CMAES_MarioGANBC_sim1_elites_freq20.csv"

#feature1Range = (0, 0.316)
#feature2Range = (0.212, 0.46)

#def compute_summary(fitnesses, num_cells, max_fitness):
#    QD_score = sum(fitnesses)
#    coverage = float(num_cells/ (gridsize * gridsize))
#    return QD_score, coverage, max_fitness


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
                    map = np.zeros((gridsize,gridsize))

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
                        cell_x = int(cell_x*gridsize)
                        cell_y = int(cell_y*gridsize)
                        if cell_x >= gridsize or cell_y >= gridsize: 
                          continue
                        map[cell_x, cell_y] = fitness
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
            coverage = float(num_cells/ (gridsize * gridsize))
            max_fitness_coverage = float(num_cells_max_fitness/ (gridsize * gridsize))

            print("QD score: " + str(QD_score))
            print("coverage: " + str(coverage))
            print("Maximum fitness coverage: " + str(max_fitness_coverage))
            return map, QD_score, coverage, max_fitness_coverage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
  
    algorithm = "Random"

    data_root = str("../test_file/test_map/" + str(algorithm))
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

    #mapT = map #np.transpose(map)
    #from IPython import embed
    #embed()
    #mapT = np.transpose(map)
    mapT = np.zeros([gridsize,gridsize])
    for i in range(0,gridsize):
        for j in range(0,gridsize):
            mapT[gridsize-1-i,j] = map[j,i]

    map = mapT
    #cmap =
    cmap = ListedColormap(sns.color_palette("coolwarm", 5).as_hex())
    with sns.axes_style("white"):
        numTicks = 10#11
        mapDims = [60,60]
        numTicksX = mapDims[0] // numTicks + 1
        numTicksY = mapDims[1] // numTicks + 1
        plt.figure(figsize=(10,9))
        sns.set(font_scale=2.5)
        #map = np.flip(map,0)
        mask = np.zeros_like(map)
        zero_indices = np.where(map == 0)
        mask[zero_indices] = np.nan
        cmap.set_bad("white") 
        g = sns.heatmap(mapT, annot=False, fmt=".0f",
                #yticklabels=[],
                vmin=0.0,
                vmax=1,
                cbar = True,
                mask = mask,
                cmap = cmap,
                rasterized=True)
                #vmin=np.nanmin(fitnessMap),
                #vmax=np.nanmax(fitnessMap))
        fig = g.get_figure()
        g.set(xticks = [0,5*60.0/5.17])
        g.set(xticklabels = ['0','5'])
        g.set(yticks = [60-4.0*60/4.17,60])
        g.set(yticklabels = ['4','0'])
        g.set(xlabel = "KL Divergence 1")
        g.set(ylabel = "KL Divergence 2")
        g.set(title = str(algorithm))

#                plt.xticks = [0,60],
#        plt.xticklabels=['0','4.17'], 
        for item in g.get_xticklabels():
            item.set_rotation(0)

        matplotlib.rcParams.update({'font.size': 20})
        plt.show()
        fig.savefig(str("KL_map_" + str(algorithm) + ".pdf"))

    #plt.figure()
    #sns.heatmap(map,yticklabels = range(0,gridsize)[::-1], annot=False, fmt=".0f", vmin=0,vmax=1)
    #plt.show()

    QD_scores_avg = np.average(QD_scores)
    coverages_avg = np.average(coverages)
    max_fitnesses_coverage_avg = np.average(max_fitnesses_coverage)
    print("Average QDscore: " + str(QD_scores_avg))
    print("Average Coverage: " + str(coverages_avg))
    print("Average Max fitness coverage: " + str(max_fitnesses_coverage_avg))

#    map, QD_scores_random, coverages_random = import_data_file(RANDOM_file_name,feature_ranges)               
#    map, QD_scores_cmame_1, coverages_cmame_1 = import_data_file(CMAME_file_name_1,feature_ranges)               
#    map, QD_scores_cmame_2, coverages_cmame_2 = import_data_file(CMAME_file_name_2,feature_ranges)               

