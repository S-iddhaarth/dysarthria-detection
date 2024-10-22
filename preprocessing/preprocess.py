from . import utils
from torch.utils.data import Dataset
import numpy as np

class piplineV1():
    def __init__(self,data:Dataset,config:dict) -> None:
        self.data = data
        self.config = config
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        dys,dys_sr,cont,cont_sr,word,integibility,speaker = self.data[idx]

        dys = utils.pre_emphasis_filter(dys,self.config["filter_coeff"])
        cont = utils.pre_emphasis_filter(cont,self.config["filter_coeff"])
        
        dys,dys_len = utils.frame(
            dys,dys_sr,frame_size=self.config["frame_size"],
            frame_stride=self.config["frame_stride"]
        )
        
        cont,cont_len = utils.frame(
            cont,cont_sr,frame_size=self.config["frame_size"],
            frame_stride=self.config["frame_stride"]
        )
        
        dys = utils.windowing(dys,dys_len,np.hamming)
        cont = utils.windowing(dys,cont_len,np.hamming)
        
        dys = utils.fourier_transform(dys,self.config['NFFT'])
        cont = utils.fourier_transform(cont,self.config['NFFT'])
        
        dys = utils.mel_filter_bank(dys,dys_sr,self.config["NFFT"],self.config["nfilt"],self.config["low_freq_mel"])
        cont = utils.mel_filter_bank(cont,cont_sr,self.config["NFFT"],self.config["nfilt"],self.config["low_freq_mel"])
        
        dys = utils.mfcc(dys)
        cont = utils.mfcc(cont)
        
        return dys,dys_sr,cont,cont_sr,word,integibility,speaker