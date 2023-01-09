from munch import Munch
import copy
from decoder.fastspeech2.fastspeech2 import FastSpeech2
from decoder.tacotron2.model import Model as TacoAR 


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model, flush=True)
    print(name,flush=True)
    print("The number of parameters: {}".format(num_params), flush=True)

def build_model(config):
    model = eval(config['model_name'], **config['model_params'])
    print_network(model, config['model_name'])

    return model
