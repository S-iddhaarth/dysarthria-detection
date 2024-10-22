import torch
import load_dataset
from torch import nn

class Trainer():
    def __init__(
        self,generator:torch.nn.Module,discriminator:torch.nn.Module,
        config:dict
        ):
        self.generator = generator
        self.discriminator = discriminator
        self.path = config["paths"]
        self.optimizer_config = config["optimizer"]
        self.mse_conf = config["MSE"]

        self.criteria = nn.MSELoss(
            size_average=self.mse_conf["size_average"], 
            reduce=self.mse_conf["reduce"], 
            reduction=self.mse_conf["reduction"]
        )

        self.optimizer_generator = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.optimizer_config['learning_rate'],
            betas=self.optimizer_config['betas'],
            eps=self.optimizer_config.get('eps', 1e-8),
            weight_decay=self.optimizer_config.get('weight_decay', 0.0),
            amsgrad=self.optimizer_config.get('amsgrad', False),
            foreach=self.optimizer_config.get('foreach', None),
            maximize=self.optimizer_config.get('maximize', False),
            capturable=self.optimizer_config.get('capturable', False),
            differentiable=self.optimizer_config.get('differentiable', False),
            fused=self.optimizer_config.get('fused', None)
        )

        self.optimizer_discriminator = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.optimizer_config['learning_rate'],
            betas=self.optimizer_config['betas'],
            eps=self.optimizer_config.get('eps', 1e-8),
            weight_decay=self.optimizer_config.get('weight_decay', 0.0),
            amsgrad=self.optimizer_config.get('amsgrad', False),
            foreach=self.optimizer_config.get('foreach', None),
            maximize=self.optimizer_config.get('maximize', False),
            capturable=self.optimizer_config.get('capturable', False),
            differentiable=self.optimizer_config.get('differentiable', False),
            fused=self.optimizer_config.get('fused', None)
        )

        self.adverserial = nn.BCELoss()
        self.train,self.valid = self._load_data()
    def _load_data(self):
        pass
    def _run_batch(self):
        self.optimizer_discriminator.zero_grad()
        self.optimizer_generator.zero_grad()
        