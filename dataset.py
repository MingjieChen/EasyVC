from torch.utils import data
import torch
import glob
import os
from os.path import join, basename, dirname, split, exists
import numpy as np
import json
import csv
import random
import librosa
from torch.utils.data import DataLoader
from collections import defaultdict
from prosodic_encoder.ppgvc_f0.ppgvc_lf0 import get_cont_lf0 as process_ppgvc_f0
from prosodic_encoder.fastspeech2_pitch_energy.pitch_energy import process_norm_fastspeech2_pitch_energy
import decoder.vits.commons  as vits_commons
def get_dataloader(config):
    
    train_dataset = eval(config['dataset_class'])(config, config['train_meta'], config['train_set'])
    dev_dataset = eval(config['dataset_class'])(config, config['dev_meta'], config['dev_set'])
    
    if config['ngpu'] >1:
        shuffle = False
        train_sampler = torch.utils.data.distributed.DistributedSampler(
             train_dataset)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(
             dev_dataset)
    else:
        shuffle = True    
        train_sampler = None
        dev_sampler = None

    train_loader = DataLoader(
            train_dataset,
            batch_size = config['batch_size'],
            shuffle = shuffle,
            collate_fn = train_dataset.collate_fn,
            num_workers = config['num_workers'],
            sampler = train_sampler
                   
        )
    dev_loader = DataLoader(
            dev_dataset,
            batch_size = config['batch_size'],
            shuffle = False,
            collate_fn = dev_dataset.collate_fn,        
            num_workers = config['num_workers'],
            sampler = dev_sampler        
        )
    return train_loader, dev_loader

def pad_1D(inputs, length, PAD = 0):
    
    def pad_data(x, length, PAD):
        
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode = 'constant', constant_values = PAD

        )
        return x_padded
    
    max_len = max(len(x) for x in inputs)
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])    

    return padded

def pad_2D(inputs, maxlen = None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError(f'shape {x.shape[0]} excceed max_len {max_len}')
        
        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - x.shape[0]), mode = 'constant', constant_values = PAD
        )        
        return x_padded[:,:s]
    
    if maxlen:
        output = np.stack([pad(x,maxlen) for x in inputs])
    else:
        max_len = max([x.shape[0] for x in inputs])
        output = np.stack([pad(x, max_len) for x in inputs])
    return output            

            
class VITSDataset(data.Dataset):
    
    def __init__(self, config, metadata_csv, split):
        super().__init__()
        self.metadata = []
        
        # setup
        self.hop_size = config['vits_hop_size']
        self.sampling_rate = config['sampling_rate']
        self.segment_size = config['decoder_params']['segment_size'] # random slice segment size of the HIfIGAN in the VITS model.
        self.spec_max_len = config['spec_max_len']
        # read metadata
        with open(metadata_csv) as f:
            reader = csv.DictReader(f, delimiter = ',')
            for row in reader:
                # remove utterances that are too long for training.
                if config['rm_long_utt']:
                    _duration = row['duration']
                    if float(_duration) < config['max_utt_duration']:
                        self.metadata.append(row)
                else:
                    self.metadata.append(row)       
            f.close()    
        
        print(f'{split} data samples {len(self.metadata)}')
        self.batch_size = config['batch_size']
        self.drop_last = config['drop_last']
        self.sort = config['sort']
        # feature dirs
        self.spec_dir = os.path.join(config['dump_dir'], config['dataset'], split, config['mel_type'])

        self.ling_enc = config['ling_enc']
        self.ling_rep_dir = os.path.join(config['dump_dir'], config['dataset'], split, self.ling_enc)
        self.spk_enc = config['spk_enc']
        self.spk_emb_dir = os.path.join(config['dump_dir'], config['dataset'], split, self.spk_enc)
        self.pros_enc = config['pros_enc'] #e.g. ppgvc_f0
        self.pros_rep_dir = os.path.join(config['dump_dir'], config['dataset'], split, self.pros_enc)
        self.pros_rep_process_func = f'process_{self.pros_enc}'
        


                 
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata[idx]
        ID = row['ID']
        spk = row['spk']
        wav_path = row['wav_path']
        start, end = float(row['start']), float(row['end'])
        # audio
        audio, fs = librosa.load(wav_path, sr = self.sampling_rate)
        audio = audio[ int(start * self.sampling_rate):
                        int(end * self.sampling_rate)
            ]
        

        # feature path
        
        spec_path = os.path.join(self.spec_dir, spk, ID + '.npy')
        ling_rep_path = os.path.join(self.ling_rep_dir, spk, ID+'.npy')
        spk_emb_path = os.path.join(self.spk_emb_dir, spk, ID+'.npy')
        pros_rep_path = os.path.join(self.pros_rep_dir, spk, ID + '.npy')

        assert os.path.exists(spec_path), f"{spec_path}"
        assert os.path.exists(ling_rep_path), f'{ling_rep_path}'
        assert os.path.exists(spk_emb_path), f'{spk_emb_path}'
        assert os.path.exists(pros_rep_path), f'{pros_rep_path}'
        
        # load feature
        spec = np.load(spec_path)    
        spec_duration = spec.shape[0]
        
        # pad spec to match the segment_size
        spec_segment_size = self.segment_size // self.hop_size
        if spec_duration < spec_segment_size:
            spec_pad_length = spec_segment_size - spec_duration
            spec = np.pad(spec, [[0,spec_pad_length],[0,0]], mode = 'constant', constant_values = 0.)
        assert spec.shape[0] >= spec_segment_size   
        spec_duration = spec.shape[0]
        audio_duration = audio.shape[0]
        ling_rep = np.load(ling_rep_path)
        ling_duration = ling_rep.shape[0]
        spk_emb = np.load(spk_emb_path)
        pros_rep = np.load(pros_rep_path)
        pros_rep = eval(self.pros_rep_process_func)(pros_rep)
        pros_duration = pros_rep.shape[0]
        
        # up_sample ling_rep to 10hz, in case some ling_rep are 50hz or 25hz.
        factor = int(round(spec_duration / ling_duration))
        if factor > 1:
            ling_rep = np.repeat(ling_rep, factor, axis=0)
            #ling_rep = np.reshape(repeated_ling_rep, [ling_duration * factor, ling_rep.shape[1]])
            ling_duration = ling_rep.shape[0]


        # match length between spec and ling_rep
        if spec_duration > ling_duration :
            pad_vec = np.expand_dims(ling_rep[-1,:], axis = 0)
            ling_rep = np.concatenate((ling_rep, np.repeat(pad_vec, spec_duration - ling_duration, 0)),0)
                
        elif spec_duration < ling_duration:
            ling_rep = ling_rep[:spec_duration,:]
        
        # match length between spec and pros_rep
        if spec_duration > pros_duration:
            pad_vec = np.expand_dims(pros_rep[-1,:],axis = 0)
            pros_rep = np.concatenate((pros_rep, np.repeat(pad_vec, spec_duration - pros_duration, 0)),0)
        elif spec_duration < pros_duration:
            pros_rep = pros_rep[:spec_duration,:]
        
        
        # match length between audio and spec
        if audio_duration < int(spec_duration * self.hop_size):
            #pad
            pad_length = int(spec_duration * self.hop_size) - audio_duration
            audio = np.concatenate([audio, np.array([0.]*pad_length)], axis = 0)
        elif audio_duration > int(spec_duration * self.hop_size):
            audio = audio[:int(spec_duration * self.hop_size)]    
        
        # slice by spec_max_len
        if spec_duration > self.spec_max_len:
            start = random.randint(0, spec_duration - self.spec_max_len)
            end = start + self.spec_max_len
            spec_duration = self.spec_max_len
            spec = spec[start:end, :]
            ling_rep = ling_rep[start:end, :]
            pros_rep = pros_rep[start:end, :]
            audio = audio[start * self.hop_size: end *self.hop_size]


        
            
        return (audio, spec, ling_rep, pros_rep,  spk_emb, spec_duration)
    
    def collate_fn(self, data):
        # sort in batch
        batch_size = len(data)
        if self.sort:
            len_arr = np.array([d[-1] for d in data])
            idx_arr = np.argsort(~len_arr)
        else:
            idx_arr = np.arange(batch_size)    
        audio = [ data[id][0] for id in idx_arr]
        spec = [data[id][1] for id in idx_arr]
        ling_rep = [ data[id][2] for id in idx_arr]
        pros_rep = [ data[id][3] for id in idx_arr]
        spk_emb = [ data[id][4] for id in idx_arr]
        spec_length = [ data[id][5] for id in idx_arr ]
        audio_length = [len(_audio) for _audio in audio]
        
        max_spec_len = max(spec_length)
        max_wav_len = max(audio_length)
        
        padded_audio = torch.FloatTensor(pad_1D(audio, max_wav_len)).unsqueeze(1)
        padded_spec = torch.FloatTensor(pad_2D(spec, max_spec_len)).transpose(1,2)
        padded_ling_rep = torch.FloatTensor(pad_2D(ling_rep, max_spec_len)).transpose(1,2)
        padded_pros_rep = torch.FloatTensor(pad_2D(pros_rep, max_spec_len)).transpose(1,2)
        spk_emb_tensor = torch.FloatTensor(np.array(spk_emb)).unsqueeze(2)
        spec_length = torch.LongTensor(np.array(spec_length)) 
        audio_length = torch.LongTensor(np.array(audio_length))
        
        
        output = (padded_audio, padded_spec, padded_ling_rep, padded_pros_rep, spk_emb_tensor, spec_length, audio_length)
        
        return output    


            
        

class Dataset(data.Dataset):
    
    def __init__(self, config, metadata_csv, split):
        super().__init__()
        self.metadata = []
        
        # read metadata
        with open(metadata_csv) as f:
            reader = csv.DictReader(f, delimiter = ',')
            for row in reader:
                # remove utterances that are too long for training.
                if config['rm_long_utt']:
                    _duration = row['duration']
                    if float(_duration) < config['max_utt_duration']:
                        self.metadata.append(row)
                else:
                    self.metadata.append(row)        
            f.close()    
        
        print(f'{split} data samples {len(self.metadata)}')
        self.batch_size = config['batch_size']
        self.drop_last = config['drop_last']
        self.sort = config['sort']
        if 'mel_segment_length' in config:
            self.mel_segment_length = config['mel_segment_length']
        else:
            self.mel_segment_length = None    

        # feature dirs
        self.mel_dir = os.path.join(config['dump_dir'], config['dataset'], split, config['mel_type'])

        self.ling_enc = config['ling_enc']
        self.ling_rep_dir = os.path.join(config['dump_dir'], config['dataset'], split, self.ling_enc)
        self.spk_enc = config['spk_enc']
        self.spk_emb_dir = os.path.join(config['dump_dir'], config['dataset'], split, self.spk_enc)
        
        if config['pros_enc'] == 'none':
            self.pros_enc = None
            self.pros_rep_dir = None
            self.pros_rep_process_func = None
        else:    
            self.pros_enc = config['pros_enc'] #e.g. ppgvc_f0
            self.pros_rep_dir = os.path.join(config['dump_dir'], config['dataset'], split, self.pros_enc)
            self.pros_rep_process_func = f'process_{self.pros_enc}'
        # frames per step (only work for TacoMOL)
        self.frames_per_step = config['frames_per_step'] if 'frames_per_step' in config else 1



                 
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata[idx]
        ID = row['ID']
        spk = row['spk']
        
        # feature path
        mel_path = os.path.join(self.mel_dir, spk, ID + '.npy')
        
        ling_rep_path = os.path.join(self.ling_rep_dir, spk, ID+'.npy')
        spk_emb_path = os.path.join(self.spk_emb_dir, spk, ID+'.npy')

        assert os.path.exists(mel_path), f"{mel_path}"
        assert os.path.exists(ling_rep_path), f'{ling_rep_path}'
        assert os.path.exists(spk_emb_path), f'{spk_emb_path}'
        
        if self.pros_enc is not None:
            pros_rep_path = os.path.join(self.pros_rep_dir, spk, ID + '.npy')
            assert os.path.exists(pros_rep_path), f'{pros_rep_path}'
        
        # load feature
        mel = np.load(mel_path)    
        mel_duration = mel.shape[0]
        ling_rep = np.load(ling_rep_path)
        ling_duration = ling_rep.shape[0]
        spk_emb = np.load(spk_emb_path)
        if self.pros_enc is not None:
            pros_rep = np.load(pros_rep_path)
            pros_rep = eval(self.pros_rep_process_func)(pros_rep)
            pros_duration = pros_rep.shape[0]
        else:
            pros_rep = np.zeros(mel.shape)    
            pros_duration = pros_rep.shape[0]
        
        # up_sample ling_rep to 10hz, in case some ling_rep are 50hz or 25hz.
        factor = int(round(mel_duration / ling_duration))
        if factor >1 :
            ling_rep = np.repeat(ling_rep, factor, axis=0)
            #ling_rep = np.reshape(repeated_ling_rep, [ling_duration * factor, ling_rep.shape[1]])
            ling_duration = ling_rep.shape[0]


        # match length between mel and ling_rep
        if mel_duration > ling_duration :
            pad_vec = np.expand_dims(ling_rep[-1,:], axis = 0)
            ling_rep = np.concatenate((ling_rep, np.repeat(pad_vec, mel_duration - ling_duration, 0)),0)
                
        elif mel_duration < ling_duration:
            ling_rep = ling_rep[:mel_duration,:]
        
        # match length between mel and pros_rep
        if mel_duration > pros_duration:
            pad_vec = np.expand_dims(pros_rep[-1,:],axis = 0)
            pros_rep = np.concatenate((pros_rep, np.repeat(pad_vec, mel_duration - pros_duration, 0)),0)
        elif mel_duration < pros_duration:
            pros_rep = pros_rep[:mel_duration,:]
        
        
        if self.mel_segment_length is not None:
            if mel_duration > self.mel_segment_length:
                start = random.randint(0, mel_duration - self.mel_segment_length)
                end = start + self.mel_segment_length
                mel_duration = self.mel_segment_length
                mel = mel[start:end, :]
                ling_rep = ling_rep[start:end, :]
                pros_rep = pros_rep[start:end, :]


        return (mel, ling_rep, pros_rep,  spk_emb, mel_duration)
    
    def collate_fn(self, data):
        # sort in batch
        batch_size = len(data)
        if self.sort:
            len_arr = np.array([d[-1] for d in data])
            idx_arr = np.argsort(~len_arr)
        else:
            idx_arr = np.arange(batch_size)    
        mel = [ data[id][0] for id in idx_arr]
        ling_rep = [ data[id][1] for id in idx_arr]
        pros_rep = [ data[id][2] for id in idx_arr]
        spk_emb = [ data[id][3] for id in idx_arr]
        length = [ data[id][4] for id in idx_arr ]
        
        max_len = max(length)
        if max_len % self.frames_per_step != 0:
            max_len += (self.frames_per_step - max_len % self.frames_per_step)
        padded_mel = torch.FloatTensor(pad_2D(mel, max_len))
        padded_ling_rep = torch.FloatTensor(pad_2D(ling_rep, max_len))
        padded_pros_rep = torch.FloatTensor(pad_2D(pros_rep, max_len))
        spk_emb_tensor = torch.FloatTensor(np.array(spk_emb)).unsqueeze(1)
        length = torch.LongTensor(np.array(length)) 
        output = (padded_mel, padded_ling_rep, padded_pros_rep, spk_emb_tensor, length, max_len)
        
        return output    
