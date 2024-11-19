import models.GANformerRes as GAN
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
import numpy as np
import wandb
from torch.autograd import grad
from torch.nn.utils import spectral_norm
log = True
adversarial_loss = nn.BCEWithLogitsLoss()

def compute_gradient_penalty(discriminator, real_input, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    alpha = torch.rand((real_samples.size(0), 1, 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = discriminator(real_input, interpolates)
    fake = torch.ones(real_samples.size(0), 1).requires_grad_(False).to(device)
    
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class CepstralLoss(nn.Module):
    """Custom loss for cepstral coefficients"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # MSE on direct coefficients
        direct_loss = self.mse(pred, target)
        
        # Additional penalty on first few coefficients (more important for perception)
        first_coef_loss = self.mse(pred[:,:,:13], target[:,:,:13]) * 2.0
        
        # Temporal smoothness loss
        temporal_loss = self.mse(pred[:,:,1:] - pred[:,:,:-1], 
                               target[:,:,1:] - target[:,:,:-1])
        
        return direct_loss + first_coef_loss + temporal_loss

def train_step(real_input, real_output, generator, discriminator, 
              g_optimizer, d_optimizer, device='cuda', 
              n_critic=5, lambda_gp=10, lambda_l1=100):
    """Sophisticated training step with WGAN-GP loss and multiple critic updates"""
    batch_size = real_input.size(0)
    
    # Train Discriminator
    d_loss_total = 0
    for _ in range(n_critic):
        d_optimizer.zero_grad()
        
        # Generate fake samples
        with torch.no_grad():
            fake_output = generator(real_input)
            print(fake_output.shape)
            
        # Real loss
        real_validity = discriminator(real_input, real_output)
        fake_validity = discriminator(real_input, fake_output.detach())
        
        # Wasserstein loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_input, real_output, fake_output.detach(), device
        )
        
        # Total discriminator loss
        d_loss = d_loss + lambda_gp * gradient_penalty
        d_loss.backward()
        d_optimizer.step()
        d_loss_total += d_loss.item()
    
    d_loss_avg = d_loss_total / n_critic
    
    # Train Generator
    g_optimizer.zero_grad()
    
    # Generate fake samples
    fake_output = generator(real_input)
    
    # Adversarial loss
    fake_validity = discriminator(real_input, fake_output)
    g_loss_adv = -torch.mean(fake_validity)
    
    # Custom cepstral loss
    cepstral_loss = CepstralLoss()
    g_loss_cep = cepstral_loss(fake_output, real_output)
    
    # L1 loss for overall consistency
    g_loss_l1 = F.l1_loss(fake_output, real_output)
    
    # Total generator loss
    g_loss = g_loss_adv + lambda_l1 * g_loss_l1 + g_loss_cep
    g_loss.backward()
    g_optimizer.step()
    
    # Compute similarities
    with torch.no_grad():
        cosine_similarity = metrics.cosine_similarity(real_output.cpu(), fake_output.cpu())
        ncc_similarity = metrics.ncc_similarity(real_output.cpu(), fake_output.cpu())
        histogram_similarity = metrics.histogram_similarity(real_output.cpu(), fake_output.cpu())
    
    # Log metrics
    if wandb.run is not None:
        wandb.log({
            "discriminator_loss": d_loss_avg,
            "generator_loss": g_loss.item(),
            "generator_adv_loss": g_loss_adv.item(),
            "generator_cep_loss": g_loss_cep.item(),
            "generator_l1_loss": g_loss_l1.item(),
            "gradient_penalty": gradient_penalty.item(),
            "generator_lr": get_learning_rate(g_optimizer),
            "discriminator_lr": get_learning_rate(d_optimizer),
            "cosine_similarity": cosine_similarity.mean().item(),
            "ncc_similarity": ncc_similarity.mean().item(),
            "histogram_similarity": histogram_similarity.mean().item()
        })
    
    return (d_loss_avg, g_loss.item(), 
            cosine_similarity.sum(), ncc_similarity.sum(), histogram_similarity.sum())

def evaluate_step(real_input, real_output, generator, discriminator, device):
    """Evaluation step with detailed metrics"""
    batch_size = real_input.size(0)
    
    with torch.no_grad():
        # Generate fake samples
        fake_output = generator(real_input)
        
        # Compute discriminator scores
        real_validity = discriminator(real_input, real_output)
        fake_validity = discriminator(real_input, fake_output)
        
        # Compute losses
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        g_loss_adv = -torch.mean(fake_validity)
        
        # Custom cepstral loss
        cepstral_loss = CepstralLoss()
        g_loss_cep = cepstral_loss(fake_output, real_output)
        g_loss_l1 = F.l1_loss(fake_output, real_output)
        g_loss = g_loss_adv + 100 * g_loss_l1 + g_loss_cep
        
        # Compute similarities
        cosine_similarity = metrics.cosine_similarity(real_output.cpu(), fake_output.cpu())
        ncc_similarity = metrics.ncc_similarity(real_output.cpu(), fake_output.cpu())
        histogram_similarity = metrics.histogram_similarity(real_output.cpu(), fake_output.cpu())
        
        # Create visualizations
        fake = wandb.Image(fake_output[0])
        real = wandb.Image(real_output[0])
        
        # Additional cepstral-specific metrics
        mcd = compute_mel_cepstral_distortion(real_output.cpu(), fake_output.cpu())
        
        if wandb.run is not None:
            wandb.log({
                "eval/discriminator_loss": d_loss.item(),
                "eval/generator_loss": g_loss.item(),
                "eval/mel_cepstral_distortion": mcd,
                "eval/fake_sample": fake,
                "eval/real_sample": real
            })
    
    return (d_loss.item(), g_loss.item(), 
            cosine_similarity.sum(), ncc_similarity.sum(), histogram_similarity.sum(),
            fake, real, mcd)

def compute_mel_cepstral_distortion(real_cep, fake_cep):
    """Compute Mel Cepstral Distortion between real and generated coefficients"""
    # Weight matrix for MCD calculation (emphasizes lower order coefficients)
    weights = torch.exp(-0.1 * torch.arange(real_cep.size(-1)))
    
    # Compute weighted Euclidean distance
    diff = real_cep - fake_cep
    weighted_diff = diff * weights.unsqueeze(0).unsqueeze(0)
    mcd = torch.sqrt(torch.sum(weighted_diff ** 2, dim=-1)).mean()
    
    return mcd

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: optimizer, warmup_steps: int, base_scheduler: _LRScheduler, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # After warmup period, defer to the base scheduler
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            super(WarmupScheduler, self).step(epoch)
        else:
            self.base_scheduler.step(epoch)
        self.last_epoch += 1




root = 'data/UASPEECH'
img_dir = 'mfcc_low'
img_type = "reflect"
annotation = "output_low_reduced.csv"
if log:
    wandb.init(
            job_type="naive_run",
            project="The last dance",
            entity="PaneerShawarma",
            config={
                "optimizer":"sophiG",
                "scheduler":"cosine annealing",
                "warup steps":1000,
                "total steps":127*100,
                "batch size":32,
                "intelligibility":"medium"
            }
        )

dataset = data_loader.SpeechPairLoaderPreprocessed(root, img_dir, img_type, annotation,1)
total = len(dataset)
train_size = int(0.8 * total)  
valid_size = total - train_size  
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=7,
    persistent_workers=True,  
    pin_memory=True,
    prefetch_factor=4,
    shuffle=True  
)

test_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    num_workers=7,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    shuffle=False  
    )

your_channel_size = 13
generator = GAN.Generator(input_channels=80,output_channel=80).cuda()
discriminator = GAN.Discriminator(input_channels=80).cuda()


print(summary(generator,torch.rand((32,80,250)).cuda()))

g_optimizer = optimizer.SophiaG(generator.parameters(),lr=0.001)
d_optimizer = optimizer.SophiaG(discriminator.parameters(),lr=0.001)


warmup_steps = 300
total_steps = 16*300


g_scheduler_cosine = CosineAnnealingLR(g_optimizer, T_max=total_steps - warmup_steps)
d_scheduler_cosine = CosineAnnealingLR(d_optimizer, T_max=total_steps - warmup_steps)


g_scheduler = WarmupScheduler(g_optimizer, warmup_steps, g_scheduler_cosine)
d_scheduler = WarmupScheduler(d_optimizer, warmup_steps, d_scheduler_cosine)

frequency = 5
checkpoint = r'weights/experiment_3'
os.makedirs(checkpoint,exist_ok=True)
for i in range(300):
    start = time.time()
    running_train_generator_loss = 0
    running_train_discriminator_loss = 0
    running_validation_generator_loss = 0
    running_validation_discriminator_loss = 0
    running_train_cosine_similarity = 0
    running_train_ncc_similarity = 0
    running_train_histogram_similarity = 0
    running_validation_cosine_similarity = 0
    running_validation_ncc_similarity = 0
    running_validation_histogram_similarity = 0
    generator.train()
    discriminator.train()

    
    for img,out in tqdm(train_loader,total=len(train_loader),desc=f'training {i} epoch'):
        d,g,c_s,n_s,h_s = train_step(img.cuda(),out.cuda(),generator,discriminator,g_optimizer,d_optimizer)
        running_train_discriminator_loss += d
        running_train_generator_loss += g
        running_train_cosine_similarity += c_s
        running_train_ncc_similarity += n_s
        running_train_histogram_similarity += h_s
        
        

    generator.eval()
    discriminator.eval()

    for img,out in tqdm(test_loader,total=len(test_loader),desc=f'evaluating {i} epoch'):
        d,g,s_c,n_c,h_c,fake,real = evaluate_step(img.cuda(),out.cuda(),generator,discriminator)
        running_validation_discriminator_loss += d
        running_validation_generator_loss += g
        running_validation_cosine_similarity += s_c
        running_validation_ncc_similarity += n_c
        running_validation_histogram_similarity += h_c
    if log:
        wandb.log({
        "average generator trianing loss":running_train_generator_loss/len(train_loader),
        "average discriminator training loss":running_train_discriminator_loss/len(train_loader),
        "average generator validation loss":running_validation_generator_loss/len(test_loader),
        "average discriminator validation loss":running_validation_discriminator_loss/len(test_loader),
        "average training cosine similarity":running_train_cosine_similarity/train_size,
        "average training ncc similarity":running_train_ncc_similarity/train_size,
        "average training histogram similarity":running_train_histogram_similarity/train_size,
        "average validation cosine similarity":running_validation_cosine_similarity/valid_size,
        "average validation ncc similarity":running_validation_ncc_similarity/valid_size,
        "average validation histogram similarity":running_validation_histogram_similarity/valid_size,
        "fake image":fake,
        "real image":real
    })
    print(f"""-----------------------------------------------------
            Training Metrics:
            Generator Loss: {running_train_generator_loss/len(train_loader):.4f}
            Discriminator Loss: {running_train_discriminator_loss/len(train_loader):.4f}
            Cosine Similarity: {running_train_cosine_similarity/train_size:.4f}
            NCC Similarity: {running_train_ncc_similarity/train_size:.4f}
            Histogram Similarity: {running_train_histogram_similarity/train_size:.4f}

            Validation Metrics:
            Generator Loss: {running_validation_generator_loss/len(test_loader):.4f}
            Discriminator Loss: {running_validation_discriminator_loss/len(test_loader):.4f}
            Cosine Similarity: {running_validation_cosine_similarity/valid_size:.4f}
            NCC Similarity: {running_validation_ncc_similarity/valid_size:.4f}
            Histogram Similarity: {running_validation_histogram_similarity/valid_size:.4f}
            -----------------------------------------------------""")
    if (i+1)%frequency == 0:
        torch.save(generator.state_dict(),os.path.join(checkpoint,f'generator_{i+1}.pt'))
        torch.save(discriminator.state_dict(),os.path.join(checkpoint,f'discriminator_{i+1}.pt'))   