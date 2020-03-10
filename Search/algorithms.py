import numpy as np
import math
import os
import json
import toml
import pandas as pd
import csv
from util.SearchHelper import *
from util.gan_generator import *

num_params = 96
boundary_value = 1
batchSize = 3
nz = 32
RecordFrequency=20

class CMA_ES_Algorithm:

    def __init__(self, num_to_evaluate,mutation_power,population_size,feature_map,trial_name,column_names):
        self.population_size=population_size
        self.num_parents = self.population_size // 2
        self.feature_map=feature_map
        self.allRecords=pd.DataFrame(columns=column_names)
        self.mutation_power = mutation_power
        self.num_to_evaluate = num_to_evaluate
        self.individuals_evaluated = 0
        self.trial_name=trial_name

        self.mean = np.asarray([0.0] * num_params)
        self.population = []
        
        self.successful_individuals=[]

        # Setup recombination weights
        self.weights = [math.log(self.num_parents + 0.5) \
                - math.log(i+1) for i in range(self.num_parents)] 
        total_weights = sum(self.weights)
        self.weights = np.array([w/total_weights for w in self.weights])
        self.mueff = sum(self.weights) ** 2 / sum(self.weights ** 2)
	
        # Setup strategy parameters for adaptation
        self.cc = (4+self.mueff/num_params)/(num_params+4 + 2*self.mueff/num_params)
        self.cs = (self.mueff+2)/(num_params+self.mueff+5)
        self.c1 = 2/((num_params+1.3)**2+self.mueff)
        self.cmu = min(1-self.c1,2*(self.mueff-2+1/self.mueff)/((num_params+2)**2+self.mueff))
        self.damps = 1 + 2*max(0,math.sqrt((self.mueff-1)/(num_params+1))-1)+self.cs
        self.chiN = num_params**0.5 * (1-1/(4*num_params)+1./(21*num_params**2))

        # Setup evolution path variables
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(num_params)

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self,model_path):
        unscaled_params = \
            [self.mutation_power * eigenval ** 0.5 * gaussian() \
                for eigenval in self.C.eigenvalues]
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + np.array(unscaled_params)
        ind = Individual()
        ind.param_vector = unscaled_params
        #level,num_non_empty,num_enemies=gan_generate(ind.param_vector)
        level=gan_generate(ind.param_vector,batchSize,nz,model_path)
        ind.level=level
        #ind.features = (num_non_empty, num_enemies)
        return ind
    
    def return_evaluated_individual(self, ind):
        ind.make_features()
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.allRecords.loc[ind.ID]=["CMA-ES"]+[ind.param_vector]+ind.statsList+list(ind.features)

        if self.individuals_evaluated % RecordFrequency == 0:

            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open("logs/"+self.trial_name+"_elites_freq"+str(RecordFrequency)+".csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close()

        self.population.append(ind)
        if len(self.population) < self.population_size:
            return

        for cur in self.population:
            self.feature_map.add(cur)

        # Sort by fitness
        parents = sorted(self.population, key=lambda x: x.fitness)[::-1]
        parents = parents[:self.num_parents]
        #print('----', parents[0].fitness)

        # Recombination of the new mean
        old_mean = self.mean
        self.mean = sum(ind.param_vector * w for ind, w in zip(parents, self.weights))
        #print('mean', self.mean)

        # Update the evolution path
        y = self.mean - old_mean
        z = np.matmul(self.C.invsqrt, y)
        self.ps = (1-self.cs) * self.ps +\
           (math.sqrt(self.cs * (2 - self.cs) * self.mueff) / self.mutation_power) * z
        left = sum(x**2 for x in self.ps) / num_params \
             / (1-(1-self.cs)**(2*self.individuals_evaluated / self.population_size)) 
        right = 2 + 4./(num_params+1)
        hsig = 1 if left < right else 0

        self.pc = (1-self.cc) * self.pc + \
            hsig * math.sqrt(self.cc*(2-self.cc)*self.mueff) * y

        # Adapt the covariance matrix
        c1a = self.c1 * (1 - (1-hsig**2) * self.cc * (2 - self.cc))
        self.C.C *= (1 - c1a - self.cmu)
        self.C.C += self.c1 * np.outer(self.pc, self.pc)
        for k, w in enumerate(self.weights):
            dv = parents[k].param_vector - old_mean
            self.C.C += w * self.cmu * np.outer(dv, dv) / (self.mutation_power ** 2)
        #print('C', np.amin(self.C.C))

        # Updated the covariance matrix decomposition and inverse
        self.C.update_eigensystem()
	
        # Update sigma
        cn, sum_square_ps = self.cs / self.damps, sum(x**2 for x in self.ps)
        self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))


        # Reset the population
        self.population.clear()


class CMA_ME_Algorithm:

    def __init__(self, mutation_power, num_to_evaluate, population_size,feature_map,trial_name,column_names):     
        self.allRecords=pd.DataFrame(columns=column_names)
        self.population_size=population_size
        self.sigma = mutation_power
        self.num_released = 0
        self.population = []
        self.feature_map = feature_map
        self.num_features = len(self.feature_map.feature_ranges)
        self.num_to_evaluate = num_to_evaluate
        self.individuals_evaluated = 0
        self.trial_name=trial_name
        self.reset()
        

    def reset(self):
        self.mutation_power = self.sigma
        if len(self.feature_map.elite_map) == 0:
            self.mean = np.asarray([0.0] * num_params)
        else:
            self.mean = self.feature_map.get_random_elite().param_vector  
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)
        self.C = DecompMatrix(num_params)
    
    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True

        return False

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self,model_path):
        unscaled_params = \
            [self.mutation_power * eigenval ** 0.5 * gaussian() \
                for eigenval in self.C.eigenvalues]
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + np.array(unscaled_params)
        ind = Individual()
        ind.param_vector = unscaled_params
        level=gan_generate(ind.param_vector,batchSize,nz,model_path)
        ind.level=level

        return ind

    def return_evaluated_individual(self, ind):
        ind.make_features()
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.allRecords.loc[ind.ID]=["CMA-ME-Improvement"]+[ind.param_vector]+ind.statsList+list(ind.features)
        self.population.append(ind)
        if len(self.population) < self.population_size:
            return

        # Only filter by this generation
        parents = []
        for cur in self.population:
            if self.feature_map.add(cur):
                parents.append(cur)
        num_parents = len(parents)
        needs_restart = num_parents == 0


        # Only update if there are parents
        if num_parents > 0:
            #print("did improve")
            #sys.stdout.flush()
            parents = sorted(parents, key=lambda x: x.delta)[::-1]

            # Create fresh weights for the number of elites found
            weights = [math.log(num_parents + 0.5) \
                    - math.log(i+1) for i in range(num_parents)] 
            total_weights = sum(weights)
            weights = np.array([w/total_weights for w in weights])
        
            # Dynamically update these parameters
            mueff = sum(weights) ** 2 / sum(weights ** 2)
            cc = (4+mueff/num_params)/(num_params+4 + 2*mueff/num_params)
            cs = (mueff+2)/(num_params+mueff+5)
            c1 = 2/((num_params+1.3)**2+mueff)
            cmu = min(1-c1,2*(mueff-2+1/mueff)/((num_params+2)**2+mueff))
            damps = 1 + 2*max(0,math.sqrt((mueff-1)/(num_params+1))-1)+cs
            chiN = num_params**0.5 * (1-1/(4*num_params)+1./(21*num_params**2))

            # Recombination of the new mean
            old_mean = self.mean
            self.mean = sum(ind.param_vector * w for ind, w in zip(parents, weights))

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1-cs) * self.ps +\
                (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
            left = sum(x**2 for x in self.ps) / num_params
            right = 2 + 4./(num_params+1)
            hsig = 1 if left < right else 0

            self.pc = (1-cc) * self.pc + \
                hsig * math.sqrt(cc*(2-cc)*mueff) * y

            # Adapt the covariance matrix
            c1a = c1 * (1 - (1-hsig**2) * cc * (2 - cc))
            self.C.C *= (1 - c1a - cmu)
            self.C.C += c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(weights):
                dv = parents[k].param_vector - old_mean
                self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power ** 2)

            # Update the covariance matrix decomposition and inverse
            if self.check_stop(parents):
                needs_restart = True
            else:
                self.C.update_eigensystem()
	
            # Update sigma
            cn, sum_square_ps = cs / damps, sum(x**2 for x in self.ps)
            self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))

        if needs_restart:
            self.reset()

        # Reset the population
        self.population.clear()
        
        if self.individuals_evaluated % RecordFrequency == 0:
            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open("logs/"+self.trial_name+"_elites_freq"+str(RecordFrequency)+".csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close()

class MapElitesAlgorithm:

    def __init__(self, mutation_power, initial_population, num_to_evaluate, feature_map,trial_name,column_names):
        self.num_to_evaluate = num_to_evaluate
        self.initial_population = initial_population
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.mutation_power = mutation_power
        self.allRecords=pd.DataFrame(columns=column_names)
        self.trial_name=trial_name

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self,model_path):
        
        ind = Individual()
        if self.individuals_evaluated < self.initial_population:
            unscaled_params = \
                [np.random.normal(0,1) for _ in range(num_params)]
            ind.param_vector = unscaled_params
        else:
            parent = self.feature_map.get_random_elite()
            unscaled_params = \
                [parent.param_vector[i] + self.mutation_power * gaussian() for i in range(num_params)]
            ind.param_vector = unscaled_params

        level=gan_generate(ind.param_vector,batchSize,nz,model_path)
        ind.level=level
 
        return ind

    def return_evaluated_individual(self, ind):

        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)
        self.allRecords.loc[ind.ID]=["MAP-Elite"]+[ind.param_vector]+ind.statsList+list(ind.features)

        #print("Evaluated One Individual")
    
        if self.individuals_evaluated % RecordFrequency == 0:
            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open("logs/"+self.trial_name+"_elites_freq"+str(RecordFrequency)+".csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close()

class ISOLineDDAlgorithm:

    def __init__(self, mutation_power1,mutation_power2, initial_population, num_to_evaluate, feature_map,trial_name,column_names):
        self.num_to_evaluate = num_to_evaluate
        self.initial_population = initial_population
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.mutation_power1 = mutation_power1
        self.mutation_power2=mutation_power2
        self.allRecords=pd.DataFrame(columns=column_names)
        self.trial_name=trial_name

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self,model_path):
        
        ind = Individual()
        if self.individuals_evaluated < self.initial_population:
            unscaled_params = \
                [np.random.normal(0,1) for _ in range(num_params)]
            ind.param_vector = unscaled_params
        else:
            pos = random.randint(0, len(self.feature_map.elite_indices)-1)
            index1 = self.feature_map.elite_indices[pos]
            pos = random.randint(0, len(self.feature_map.elite_indices)-1)
            index2 = self.feature_map.elite_indices[pos]
            
            parent1=self.feature_map.elite_map[index1]
            parent2=self.feature_map.elite_map[index2]

            unscaled_params = \
                [parent1.param_vector[i] + self.mutation_power1 * gaussian() + self.mutation_power2 * (parent1.param_vector[i]-parent2.param_vector[i]) * gaussian() for i in range(num_params)]
            ind.param_vector = unscaled_params
        level=gan_generate(ind.param_vector,batchSize,nz,model_path)
        ind.level=level
        return ind

    def return_evaluated_individual(self, ind):

        ind.make_features() #now this does nothing
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        
        self.feature_map.add(ind)
        self.allRecords.loc[ind.ID]=["ISOLine-DD"]+[ind.param_vector]+ind.statsList+list(ind.features)

        #print("Evaluated One Individual")
       
        if self.individuals_evaluated % RecordFrequency == 0:
            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open("logs/"+self.trial_name+"_elites_freq"+str(RecordFrequency)+".csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close() 

class RandomGenerator:

    def __init__(self, num_to_evaluate, feature_map,trial_name,column_names):
        self.num_to_evaluate = num_to_evaluate
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.allRecords=pd.DataFrame(columns=column_names)
        self.trial_name=trial_name

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self,model_path):
        
        ind = Individual()
        unscaled_params = \
            [np.random.normal(0,1) for _ in range(num_params)]
        ind.param_vector = unscaled_params
       

        level=gan_generate(ind.param_vector,batchSize,nz,model_path)
        ind.level=level
 
        return ind

    def return_evaluated_individual(self, ind):

        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)
        self.allRecords.loc[ind.ID]=["Random"]+[ind.param_vector]+ind.statsList+list(ind.features)

        #print("Evaluated One Individual")
    
        if self.individuals_evaluated % RecordFrequency == 0:
            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open("logs/"+self.trial_name+"_elites_freq"+str(RecordFrequency)+".csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close()