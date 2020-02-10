import numpy
import json
import torch
from torch.autograd import Variable
import util.models.dcgan as dcgan
import toml

parsed_toml=toml.load("Searching/config/cma_me.tml")
num_params = parsed_toml["ModelParameter"]["num_params"]
boundary_value = parsed_toml["ModelParameter"]["boundary_value"]
batchSize = parsed_toml["ModelParameter"]["batchSize"]
nz = parsed_toml["ModelParameter"]["nz"]  # Dimensionality of latent vector

imageSize = parsed_toml["ModelParameter"]["imageSize"]
ngf = parsed_toml["ModelParameter"]["ngf"]
ngpu = parsed_toml["ModelParameter"]["ngpu"]
n_extra_layers = parsed_toml["ModelParameter"]["n_extra_layers"]

features = parsed_toml["ModelParameter"]["features"]
generator = dcgan.DCGAN_G(imageSize, nz, features, ngf, ngpu, n_extra_layers)
generator.load_state_dict(torch.load(parsed_toml["GANModelPath"], map_location=lambda storage, loc: storage))

def gan_generate(x,batchSize,nz):
    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1,
                                              1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    levels.data = levels.data[:, :, :16, :56]
    im = levels.data.cpu().numpy()
    im = numpy.argmax( im, axis = 1)
    return json.dumps(im[0].tolist())