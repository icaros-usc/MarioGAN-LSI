import os
import pathlib
os.environ['CLASSPATH']=os.path.join(pathlib.Path().absolute(),"Mario.jar")

from jnius import autoclass
MarioGame = autoclass('engine.core.MarioGame')
Agent = autoclass('agents.robinBaumgarten.Agent')

import torch
import torchvision.utils as vutils
from torch.autograd import Variable

import sys
import json
import numpy
import models.dcgan as dcgan
import random
import math
import matplotlib.pyplot as plt
import os

num_params = 96
boundary_value = 5.12
batchSize = 3
nz = 32
imageSize = 64
ngf = 64
ngpu = 1
n_extra_layers = 0
features = 23

COIN = 18
GROUND = 7
ENEMY = 4
#PIPE = 6
EMPTY = 2

generator = dcgan.DCGAN_G(imageSize, nz, features, ngf, ngpu, n_extra_layers)
generator.load_state_dict(torch.load('netG_epoch_4900_5899.pth', map_location=lambda storage, loc: storage))

def get_char(x):
    return {
             0:'-',
             1:'F',
             2:'-',
             3:'y',
             4:'g',
             5:'k',
             6:'r',
             7:'X',
             8:'#',
             9:'%',
             10:'|',
             11:'t',
             12:'@',
             13:'B',
             14:'b',
             15:'?',
             16:'Q',
             17:'S',           
             18:'o',
             19:'<',
             20:'>',
             21:'[',
             22:']'
             
        }[x] 

def to_level(number_level):
    result = []
    for x in number_level:
        result.append(''.join(get_char(y) for y in x)+'\n')
    result= ''.join(result)
    return result

def gan_generate(x):
    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1,
                                              1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    levels.data = levels.data[:, :, :16, :56]
    im = levels.data.cpu().numpy()
    im = numpy.argmax( im, axis = 1)

    ground=im[:,13,:]
    above=im[:,0:12,:]
    higher=im[:,0:9,:]
    middle=im[:,10:12,:]
    num_above_ground=len(above[above!=2])
    higher_level=len(higher[higher!=2])
    middle_level=len(middle[middle!=2])
    num_ground=len(ground[ground==GROUND])
    num_ground_enemies=len(ground[ground==ENEMY])
    num_enemies =  len (im[im == ENEMY])
    num_non_empty=len(higher[higher!=2])
    #num_non_empty=num_non_empty/2 #range for each cell in x axis is 2
    return json.dumps(im[0].tolist()),num_non_empty,num_enemies

if __name__ == '__main__':
    
    fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
    level=gan_generate(fixed_noise)[0]
    print(len(level))
    output = eval(level)
    print(output)
    print(len(output))
    output = to_level(output)
    
    with open ("test.txt", "r") as myfile:
        output=''.join(myfile.readlines())

    print(output)

    agent = Agent()
    game = MarioGame()
    result = game.runGame(agent, output, 200, 0, True)
    #result = game.playGame(output, 200, 0)
    print(result)
    print("****************************************************************")
    print("Game Status: " + result.getGameStatus().toString() + " Percentage Completion: " + str(result.getCompletionPercentage()))
    print("Lives: " + str(result.getCurrentLives()) + " Coins: " + str(result.getCurrentCoins()) +
          " Remaining Time: " + str(int(math.ceil(result.getRemainingTime() / 1000.0))))
    #print("Mario State: " + result.getMarioMode() + " (Mushrooms: " + result.getNumCollectedMushrooms() + " Fire Flowers: " + result.getNumCollectedFireflower() + ")")
    #print("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp() +
    #      " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() +
    #      " Falls: " + result.getKillsByFall() + ")")
    #print("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps() +
    #      " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime())
    print("****************************************************************")
