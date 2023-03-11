from munch import Munch
import copy
from decoder.fastspeech2.fastspeech2 import FastSpeech2
from decoder.taco_ar.model import Model as TacoAR 
from decoder.taco_mol.model import MelDecoderMOLv2 as TacoMOL
from decoder.vits.models import VITS
from decoder.grad_tts.grad_tts_model import GradTTS
from decoder.diffwave.model import DiffWave


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model, flush=True)
    print(name,flush=True)
    print("The number of parameters: {}".format(num_params), flush=True)

def build_model(config):
    decoder_params = config['decoder_params']
    model = eval(config['decoder'])(config = decoder_params)
    print_network(model, config['decoder'])

    return model
