import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter


file_name="MAPELITES_KLBC_sim10_elites_freq20.csv"
file_name_list=file_name.split("_")
with open("logs/"+file_name, newline='') as csvfile:
        all_records = csv.reader(csvfile, delimiter=',')
        for i,one_map in enumerate(all_records):
            if(i%2==1):
                continue
            plt.figure()
            num_simulations=((i/2)+1)*20
            #print(one_map)
            x=[]
            y=[]
            fitness=[]
            i=0
            for data_point in one_map:
                i=i+1
                data_point=data_point[1:-1]
                data_point_info=data_point.split(', ')
                x.append(float(data_point_info[2]))
                y.append(float(data_point_info[3]))
                fitness.append(float(data_point_info[1][1:-1]))
                
            gridsize=60
            plt.hexbin(x, y, C=fitness, gridsize=gridsize, cmap=cm.jet, bins=None)
            #plt.axis([x.min(), x.max(), y.min(), y.max()])
            
            cb = plt.colorbar()
            cb.set_label('fitness')
            plt.savefig("Visualize/"+file_name_list[0]+"_"+file_name_list[0]+"_"+str(num_simulations)+"_simulations.jpg")
            #plt.show()   
            plt.close()