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

batchSize = 32
num_params = 96
boundary_value = 1
nz = 32

imageSize = 64
ngf = 64
ngpu = 1
n_extra_layers = 0
model_path="GANTrain/samples/netG_epoch_4999_7684.pth"
features = len(json.load(open('GANTrain/index2str.json')))

generator = dcgan.DCGAN_G(imageSize, nz, features, ngf, ngpu, n_extra_layers)
generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

with open('GANTrain/index2str.json') as f:
  index2str = json.load(f)
  
def get_char(x):
    return index2str[str(x)] 

def to_level(number_level):
    result = []
    number_level=eval(number_level)
    for x in number_level:
        #print(x)
        result.append(''.join(get_char(y) for y in x)+'\n')
    result= ''.join(result)
    return result

def gan_generate(x,batchSize,nz):
    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1,
                                              1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    levels.data = levels.data[:, :, :16, :56]
    im = levels.data.cpu().numpy()
    im = numpy.argmax( im, axis = 1)
    return json.dumps(im[0].tolist())

if __name__ == '__main__':
    
    fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
    level=gan_generate(fixed_noise,32,32)
    
    #comment the below 2 lines if want to generate levels randomly
    #with open ("GANTrain/samples/fake_level_epoch_999_8815.json") as f:
    #    level=json.dumps(json.load(f))
    
    realLevel=to_level(level)

    realLevel=open("test_level.txt").read()

    agent = Agent()
    game = MarioGame()
    result = game.runGame(agent, realLevel, 20, 0,True)
    from IPython import embed
    embed()
    print(realLevel)
    #result = game.playGame(realLevel, 200, 0)
    #print(result)
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
