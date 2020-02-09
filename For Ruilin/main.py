from __future__ import print_function

import argparse
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import dcgan
import os
#os.chdir("./src/GANTraining")
# print(os.getcwd())
# Run with "python main.py"

def get_lvls(folder):
    lvls = []
    #files = [f for f in os.listdir(folder) if 'lvl' in f]
    files = sorted([f for f in os.listdir(folder) if 'lvl' in f])
    for f in files:
        f=open(os.path.join(folder, f))
        lvl=f.readlines()
        clean_level = []
        for l in lvl:
            if len(l.strip()) > 0:
                clean_level.append(l.strip())
        lvls.append(clean_level)
    return lvls

def build_indeces(lvls, same=None):
    full_string = ''
    for lvl in lvls:
        full_string += ''.join(lvl)
    uniqueSymbols = set(full_string)
    str2index = {}
    index2str = {}
    c_index = 0
    if same is not None:
        for g in same:
            for c in g:
                str2index[c] = c_index
            index2str[c_index] = g[0]
            c_index += 1
    for symbol in uniqueSymbols:
        if symbol not in str2index:
            str2index[symbol] = c_index
            index2str[c_index] = symbol
            c_index += 1
    return str2index, index2str, c_index

def get_windows(lvls, width, indeces):
    result = []
    for lvl in lvls:
        for shift in range(len(lvl[0]) - width + 1):
            window = np.zeros((len(lvl), width))
            for y in range(len(lvl)):
                for x in range(width):
                    c = lvl[y][x+shift]
                    window[y][x] = indeces[c]
            result.append(window)
    return np.array(result).astype(np.uint8)

def get_windows_from_folder(folder, width=56, same=None):
    lvls = get_lvls(folder)
    str2index, index2str, types = build_indeces(lvls, same)
    windows = get_windows(lvls, width, str2index)
    return windows, types, index2str

normal_sames = ['*bB', 'gE', '!Q', '@?']
compressed_sames = ['-12', '*bB', 'yY', 'gGE', 'kK', 'rR', 'tT', 'SCUL', '!Q', '@?']

def bool2arr(v):
    if isinstance(v, bool):
       return [normal_sames, compressed_sames][v]
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return compressed_sames
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return normal_sames
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--problem', type=int, default=0, help='Level examples')
parser.add_argument('--json', default=None, help='Json file')
parser.add_argument('--lvls', default=None, help='Levels Folder')
parser.add_argument('--compress', type=bool2arr, default=compressed_sames, help='True to use the compressed representation')
parser.add_argument('--wwidth', type=int, default=56, help='The window width size since the height is fixed')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system(f'mkdir {opt.experiment}')


if opt.seed == 0:
    opt.seed = random.randint(1, 10000)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

map_size = 64

if opt.lvls is None:
    if opt.json is None:
        X, z_dims, index2str = get_windows_from_folder('original', opt.wwidth, opt.compress)
    else:
        examplesJson = opt.json
        X = np.array(json.load(open(examplesJson)))
        z_dims = 23 # Number different tile types
else:
    X, z_dims, index2str = get_windows_from_folder(opt.lvls, opt.wwidth, opt.compress)

num_batches = X.shape[0] / opt.batchSize
print("Amy Print Start")
print("num_batches: " + str(num_batches))
print("X.shape[0]: " + str(X.shape[0]))
print("BatchSize: " + str(opt.batchSize))
print("Amy Print End")


print("SHAPE ", X.shape)
X_onehot = np.eye(z_dims, dtype='uint8')[X]

X_onehot = np.rollaxis(X_onehot, 3, 1)
print("SHAPE ", X_onehot.shape)  # (173, 14, 28, 16)

X_train = np.zeros((X.shape[0], z_dims, map_size, map_size)) * 2

X_train[:, 2, :, :] = 1.0  # Fill with empty space

# Pad part of level so its a square
X_train[:X.shape[0], :, :X.shape[1], :X.shape[2]] = X_onehot

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

n_extra_layers = int(opt.n_extra_layers)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '':  # load checkpoint if needed
    print(os. getcwd())
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = dcgan.DCGAN_D(map_size, nz, z_dims, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1


def tiles2image(tiles):
    return plt.get_cmap('rainbow')(tiles / float(z_dims))


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return image


if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    print("Using ADAM")
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

gen_iterations = 0
for epoch in range(opt.niter):

    # ! data_iter = iter(dataloader)

    X_train = X_train[torch.randperm(len(X_train))]

    i = 0
    while i < num_batches:  # len(dataloader):

        print(f"while i ({i}) < num_batches ({num_batches})")
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
            print("1. Set Diters: " + str(Diters))
        else:
            Diters = opt.Diters
            print("2. Set Diters from Command Line: " + str(Diters))

        j = 0
        while j < Diters and i < num_batches:  # len(dataloader):

            if(j == 0 or j == Diters - 1 or i == 0 or i == num_batches - 1 or i == 65 or j == 65):
                print(f"    while j ({j}) < Diters ({Diters}) and i ({i}) < num_batches ({num_batches})")
                print(f"         Train Discriminator: d-iterations (j = {j}), number of batches (i = {i})")
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data = X_train[i * opt.batchSize:(i + 1) * opt.batchSize]

            i += 1

            real_cpu = torch.FloatTensor(data)

            if (False):
                # im = data.cpu().numpy()
                print(data.shape)
                real_cpu = combine_images(tiles2image(np.argmax(data, axis=1)))
                print(real_cpu)
                plt.imsave('{0}/real_samples.png'.format(opt.experiment), real_cpu)
                exit()

            netD.zero_grad()
            # batch_size = num_samples #real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = netD(inputv)
            errD_real.backward(one)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            with torch.no_grad():
            	#noisev = Variable(noise, volatile=True)  # totally freeze netG
            	noisev = Variable(noise)  # totally freeze netG
            fake = Variable(netG(noisev).data)
            inputv = fake
            errD_fake = netD(inputv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()
           

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        errG = netD(fake)
        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1
        print(f"         Train Generator: (gen_iterations = {gen_iterations})")
        print()

        print("[current-epoch / total epochs][place-in-batch / total-batches]")
        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (epoch, opt.niter, i, num_batches, gen_iterations,
                 errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        if gen_iterations % 100 == 0 or epoch == opt.niter - 1:  # was 500

            with torch.no_grad():
            	#fake = netG(Variable(fixed_noise, volatile=True))
            	fake = netG(Variable(fixed_noise))

            im = fake.data.cpu().numpy()
            # print('SHAPE fake',type(im), im.shape)
            # print('SUM ',np.sum( im, axis = 1) )

            im = combine_images(tiles2image(np.argmax(im, axis=1)))

            #plt.imsave('{0}/mario_fake_samples_{1}_{2}.png'.format(opt.experiment, gen_iterations, opt.seed), im)
            plt.imsave(F'{opt.experiment}/mario_fake_samples_[{epoch}-{opt.niter}][{i}-{num_batches}][{gen_iterations}]_seed-{opt.seed}.png', im)

            torch.save(netG.state_dict(), f'{opt.experiment}/netG_epoch_[{epoch}-{opt.niter}][{i}-{num_batches}][{gen_iterations}]-seed-{opt.seed}.pth')
            torch.save(netD.state_dict(), f'{opt.experiment}/netD_end-of-epoch_[{epoch}-{opt.niter}][{i}-{num_batches}][{gen_iterations}]-seed-{opt.seed}.pth')
            #torch.save(netD.state_dict(), f'{opt.experiment}/netD_epoch_[{epoch}-{opt.niter}][{i}-{num_batches}][{gen_iterations}]-seed-{opt.seed}.pth')
            #torch.save(netD.state_dict(), f'{opt.experiment}/netD_epoch_[{epoch}-{opt.niter}][{i}-{num_batches}][{gen_iterations}]-seed-{opt.seed}.pth')

            #torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_{2}.pth'.format(opt.experiment, gen_iterations, opt.seed))
            #torch.save(netD.state_dict(), '{0}/netD_epoch_{1}_{2}.pth'.format(opt.experiment, gen_iterations, opt.seed))

    # do checkpointing
    #if(epoch % 50 == 0):
        #torch.save(netG.state_dict(), f'{opt.experiment}/netG_end-of-epoch_[{epoch}-{opt.niter}][{i}-{num_batches}][{gen_iterations}]-seed-{opt.seed}.pth')
     #   torch.save(netD.state_dict(), f'{opt.experiment}/netD_end-of-epoch_[{epoch}-{opt.niter}][{i}-{num_batches}][{gen_iterations}]-seed-{opt.seed}.pth')
    ####################################################################
    #Amy - Removed checkpoints at every epoch. Fucking enormous outputs
    ####################################################################

    #torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_{2}.pth'.format(opt.experiment, epoch, opt.seed))
    #torch.save(netD.state_dict(), '{0}/netD_epoch_{1}_{2}.pth'.format(opt.experiment, epoch, opt.seed))
    




    # torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    # torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
