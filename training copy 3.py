import models.GANformerResInv as GAN
import data_loader
from torch.utils.data import DataLoader,random_split
import wandb
import optimizer
import similarity_metrics.metrics as metrics
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
import torch
from tqdm import tqdm
from modelsummary import summary
torch.backends.cudnn.benchmark = True


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import wandb
from torch.utils.data import DataLoader, random_split
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import transforms

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_scheduler, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Cosine warmup instead of linear
            warmup_factor = 0.5 * (1 + np.cos(np.pi * (1 - float(self.last_epoch) / float(max(1, self.warmup_steps)))))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            super(WarmupScheduler, self).step(epoch)
        else:
            self.base_scheduler.step(epoch)
        self.last_epoch += 1

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data + 
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

def train_step(real_input, real_output, generator, discriminator, 
               g_optimizer, d_optimizer, scaler, ema,
               adversarial_loss, l1_loss):
    batch_size = real_input.size(0)
    real_label = torch.ones(batch_size, 1, device=real_input.device)
    fake_label = torch.zeros(batch_size, 1, device=real_input.device)
    
    # Train Discriminator
    d_optimizer.zero_grad()
    
    with autocast():
        fake_output = generator(real_input)
        
        # Add noise to labels for label smoothing
        real_label = real_label - 0.1 * torch.rand_like(real_label)
        fake_label = fake_label + 0.1 * torch.rand_like(fake_label)
        
        real_loss = adversarial_loss(discriminator(real_input, real_output), real_label)
        fake_loss = adversarial_loss(discriminator(real_input, fake_output.detach()), fake_label)
        
        d_loss = (real_loss + fake_loss) / 2
    
    scaler.scale(d_loss).backward()
    scaler.unscale_(d_optimizer)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    scaler.step(d_optimizer)
    
    # Train Generator
    g_optimizer.zero_grad()
    
    with autocast():
        g_loss_gan = adversarial_loss(discriminator(real_input, fake_output), real_label)
        g_loss_l1 = l1_loss(fake_output, real_output)
        
        # Feature matching loss
        with torch.no_grad():
            real_features = discriminator.get_features(real_input, real_output)
        fake_features = discriminator.get_features(real_input, fake_output)
        feature_loss = sum(F.l1_loss(f, r) for f, r in zip(fake_features, real_features))
        
        # Total generator loss with weighted components
        g_loss = g_loss_gan + 100 * g_loss_l1 + 10 * feature_loss
    
    scaler.scale(g_loss).backward()
    scaler.unscale_(g_optimizer)
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    scaler.step(g_optimizer)
    scaler.update()
    
    # Update EMA model
    ema.update()
    
    # Calculate metrics
    with torch.no_grad():
        cosine_similarity = metrics.cosine_similarity(real_output.cpu(), fake_output.cpu())
        ncc_similarity = metrics.ncc_similarity(real_output.cpu(), fake_output.cpu())
        histogram_similarity = metrics.histogram_similarity(real_output.cpu(), fake_output.cpu())
    
    if wandb.run is not None:
        wandb.log({
            "discriminator_loss": d_loss.item(),
            "generator_loss": g_loss.item(),
            "generator_lr": g_optimizer.param_groups[0]['lr'],
            "discriminator_lr": d_optimizer.param_groups[0]['lr'],
            "feature_matching_loss": feature_loss.item(),
        })
    
    return (d_loss.item(), g_loss.item(), 
            cosine_similarity.sum(), ncc_similarity.sum(), 
            histogram_similarity.sum())

def main():
    # Hyperparameters
    config = {
        "batch_size": 64,  # Reduced from 128 for better stability
        "num_epochs": 300,
        "warmup_steps": 1000,
        "g_lr": 2e-4,  # Adjusted learning rate
        "d_lr": 1e-4,  # Discriminator learns slower than generator
        "betas": (0.5, 0.999),  # Better stability for GANs
        "ema_decay": 0.999,
        "gradient_clip": 1.0,
    }
    
    if wandb.run is not None:
        wandb.init(
            project="The last dance",
            entity="PaneerShawarma",
            config=config
        )
    
    # Dataset setup with better augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.3),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = data_loader.SpeechPairLoaderPreprocessed(
        root='data/UASPEECH',
        img_dir='mfcc_low',
        img_type="reflect_interpolate",
        annotation="output_low_reduced.csv",select=1,transform=None
    )
    
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=7,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=True
    )
    
    # Model setup
    generator = GAN.Generator().cuda()
    discriminator = GAN.Discriminator().cuda()
    
    # Initialize EMA
    ema = EMA(generator, decay=config["ema_decay"])
    
    # Optimizers
    g_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config["g_lr"],
        betas=config["betas"],
        weight_decay=0.01
    )
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config["d_lr"],
        betas=config["betas"],
        weight_decay=0.01
    )
    
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Schedulers
    total_steps = len(train_loader) * config["num_epochs"]
    g_scheduler_cosine = CosineAnnealingLR(
        g_optimizer,
        T_max=total_steps - config["warmup_steps"]
    )
    d_scheduler_cosine = CosineAnnealingLR(
        d_optimizer,
        T_max=total_steps - config["warmup_steps"]
    )
    
    g_scheduler = WarmupScheduler(
        g_optimizer,
        config["warmup_steps"],
        g_scheduler_cosine
    )
    d_scheduler = WarmupScheduler(
        d_optimizer,
        config["warmup_steps"],
        d_scheduler_cosine
    )
    
    # Initialize mixed precision training
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        generator.train()
        discriminator.train()
        
        for batch_idx, (real_input, real_output) in enumerate(train_loader):
            real_input, real_output = real_input.cuda(), real_output.cuda()
            
            metrics = train_step(
                real_input, real_output,
                generator, discriminator,
                g_optimizer, d_optimizer,
                scaler, ema,
                adversarial_loss, l1_loss
            )
            
            g_scheduler.step()
            d_scheduler.step()
            
        # Validation at end of epoch
        if epoch % 5 == 0:
            ema.apply_shadow()
            validate(valid_dataset, generator, discriminator)
            ema.restore()

if __name__ == "__main__":
    main()