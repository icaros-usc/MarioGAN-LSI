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


gridsize = 60
rows = []

def import_data_file(file_name,feature_ranges, algorithm_name, trial_indx):

    with open(file_name) as csvfile:
            all_records = csv.reader(csvfile, delimiter=',')
            QD_scores = []
            inds_evaluated = 0  

            print("importing data from file: " + str(file_name))
            for i,one_map in enumerate(all_records):
                #if i == 499:
                #    map = np.zeros((gridsize,gridsize))
                   # print(i)
                    fitnesses = []
                    num_cells = 0
 
                    inds_evaluated = inds_evaluated + 20  
                    for data_point in one_map: 
                        #i=i+1
                        data_point=data_point[1:-1]
                        feature1Range = feature_ranges[0]
                        feature2Range = feature_ranges[1]
                        data_point_info=data_point.split(', ')
                        #from IPython import embed
                        #embed()
                        if data_point_info == ['']:
                            break
                        #bc_1 = float(data_point_info[2])
                        #bc_2 = float(data_point_info[3])
                        #cell_x =(bc_1 - feature1Range[0])/(feature1Range[1]-feature1Range[0])
                        #cell_y = (bc_2 - feature2Range[0])/(feature2Range[1]-feature2Range[0])
                        
                        #from IPython import embed
                        #embed()

                        fitness = float(data_point_info[1][1:-1])
#                        if fitness > max_fitness: 
#                            max_fitness = fitness

                        #cell_x = gridsize-1 - int(cell_x*gridsize)
                        #cell_x = int(cell_x*gridsize)
                        #cell_y = int(cell_y*gridsize)
                        #if cell_x >= gridsize or cell_y >= gridsize: 
                        #  continue
                        #map[cell_x, cell_y] = fitness
                        #cell_indx = int(data_point_info[0])
                        fitnesses.append(fitness)
                        num_cells = num_cells + 1
                    QD_score = sum(fitnesses)
                    #print("QD score is: " + str(QD_score))
                    QD_scores.append(QD_score)
                    coverage = float(num_cells/ (2**8))

                    rows.append([algorithm_name, trial_indx, inds_evaluated, coverage, QD_score])
            return QD_scores

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
    
    rows.append(['Algorithm', 'Trial', 'Evaluations', 'Cells Occupied', 'QD-Score'])

    #data_root = 'test_file'
    data_root = '../logs_12/8Binary/CMAME'
    files = sorted([f for f in os.listdir(data_root)])[0:]
    #from IPython import embed
    #embed()
    trial_indx = 1
    for file_name in files:
    #file_name = files[1]
      file_name = str(data_root + "/" + file_name)
      QD_score_file = import_data_file(file_name,feature_ranges, 'CMA-ME', trial_indx)
      trial_indx = trial_indx + 1

    #data_root = 'test_file'
    data_root = '../logs_12/8Binary/ME'
    files = sorted([f for f in os.listdir(data_root)])[0:]
    #from IPython import embed
    #embed()
    trial_indx = 1
    for file_name in files:
    #file_name = files[1]
      file_name = str(data_root + "/" + file_name)
      QD_score_file = import_data_file(file_name,feature_ranges, 'MAP-Elites', trial_indx)
      trial_indx = trial_indx + 1

    #data_root = 'test_file'
    data_root = '../logs_12/8Binary/MEline'
    files = sorted([f for f in os.listdir(data_root)])[0:]
    #from IPython import embed
    #embed()
    trial_indx = 1
    for file_name in files:
    #file_name = files[1]
      file_name = str(data_root + "/" + file_name)
      QD_score_file = import_data_file(file_name,feature_ranges, 'ME (line)', trial_indx)
      trial_indx = trial_indx + 1


    data_root = '../logs_12/8Binary/Random'
    files = sorted([f for f in os.listdir(data_root)])[0:]
    #from IPython import embed
    #embed()
    trial_indx = 1
    for file_name in files:
    #file_name = files[1]
      file_name = str(data_root + "/" + file_name)
      QD_score_file = import_data_file(file_name,feature_ranges, 'Random', trial_indx)
      trial_indx = trial_indx + 1

    data_root = '../logs_12/8Binary/CMAES'
    files = sorted([f for f in os.listdir(data_root)])[0:]
    #from IPython import embed
    #embed()
    trial_indx = 1
    for file_name in files:
    #file_name = files[1]
      file_name = str(data_root + "/" + file_name)
      QD_score_file = import_data_file(file_name,feature_ranges, 'CMA-ES', trial_indx)
      trial_indx = trial_indx + 1
    summary_filename = '../logs_12/8Binary/summary.csv'
    with open(summary_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for r in rows:
           csv_writer.writerow(r)

    # import seaborn as sns; sns.set()

    # import matplotlib.pyplot as plt

    # fmri = sns.load_dataset("fmri")

    # plt.figure()
    # ax = sns.lineplot(x="timepoint", y="signal", data=fmri)
    # plt.show()
       
    #from IPython import embed
    #embed()
    #QD_scores_avg = np.average(QD_scores)
    #coverages_avg = np.average(coverages)
    #max_fitnesses_coverage_avg = np.average(max_fitnesses_coverage)
    #print("Average QDscore: " + str(QD_scores_avg))
    #print("Average Coverage: " + str(coverages_avg))
    #print("Average Max fitness coverage: " + str(max_fitnesses_coverage_avg))

#    map, QD_scores_random, coverages_random = import_data_file(RANDOM_file_name,feature_ranges)               
#    map, QD_scores_cmame_1, coverages_cmame_1 = import_data_file(CMAME_file_name_1,feature_ranges)               
#    map, QD_scores_cmame_2, coverages_cmame_2 = import_data_file(CMAME_file_name_2,feature_ranges)               

