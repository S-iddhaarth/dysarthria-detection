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

log = True
adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

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

def train_step(real_input, real_output, generator, discriminator, g_optimizer, d_optimizer):
    batch_size = real_input.size(0)
    real_label = torch.ones(batch_size, 1).cuda()
    fake_label = torch.zeros(batch_size, 1).cuda()
    d_optimizer.zero_grad()
    
    fake_output = generator(real_input)
    
    real_loss = adversarial_loss(discriminator(real_input, real_output), real_label)
    
    fake_loss = adversarial_loss(discriminator(real_input, fake_output.detach()), fake_label)
    
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    d_optimizer.step()
    
    g_optimizer.zero_grad()
    
    g_loss_gan = adversarial_loss(discriminator(real_input, fake_output), real_label)
    
    g_loss_l1 = l1_loss(fake_output, real_output)
    
    g_loss = g_loss_gan + 100 * g_loss_l1 
    g_loss.backward()
    g_optimizer.step()
    
    g_scheduler.step()
    d_scheduler.step()
    cosine_similarity = metrics.cosine_similarity(real_output.detach().cpu(),fake_output.detach().cpu())
    ncc_imilarity = metrics.ncc_similarity(real_output.detach().cpu(),fake_output.detach().cpu())
    histogram_similarity = metrics.histogram_similarity(real_output.detach().cpu(),fake_output.detach().cpu())
    
    lr_g = get_learning_rate(g_optimizer)
    lr_d = get_learning_rate(d_optimizer)
    if log:
        wandb.log({
        "discriminator loss per step":d_loss.item(),
        "generator loss per step":g_loss.item(),
        "generator lr":lr_g,
        "discriminator lr":lr_d
    })
    
    return d_loss.item(), g_loss.item(),cosine_similarity.sum(),ncc_imilarity.sum(),histogram_similarity.sum()

def evaluate_step(real_input, real_output, generator, discriminator):
    batch_size = real_input.size(0)
    real_label = torch.ones(batch_size, 1).cuda()
    fake_label = torch.zeros(batch_size, 1).cuda()

    with torch.no_grad():
        fake_output = generator(real_input)

        real_loss = adversarial_loss(discriminator(real_input, real_output), real_label)
        fake_loss = adversarial_loss(discriminator(real_input, fake_output), fake_label)
        d_loss = (real_loss + fake_loss) / 2

        g_loss_gan = adversarial_loss(discriminator(real_input, fake_output), real_label)
        g_loss_l1 = l1_loss(fake_output, real_output)
        g_loss = g_loss_gan + 100 * g_loss_l1
        
        cosine_similarity = metrics.cosine_similarity(real_output.detach().cpu(),fake_output.detach().cpu())
        ncc_imilarity = metrics.ncc_similarity(real_output.detach().cpu(),fake_output.detach().cpu())
        histogram_similarity = metrics.histogram_similarity(real_output.detach().cpu(),fake_output.detach().cpu())
        fake = wandb.Image(fake_output[0])
        real = wandb.Image(real_output[0])

    return d_loss.item(), g_loss.item(),cosine_similarity.sum(),ncc_imilarity.sum(),histogram_similarity.sum(),fake,real



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
    batch_size=128,
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

g_optimizer = torch.optim.AdamW(generator.parameters(),lr=0.01)
d_optimizer = torch.optim.AdamW(discriminator.parameters(),lr=0.01)


warmup_steps = 100
total_steps = 16*100


g_scheduler_cosine = CosineAnnealingLR(g_optimizer, T_max=total_steps - warmup_steps)
d_scheduler_cosine = CosineAnnealingLR(d_optimizer, T_max=total_steps - warmup_steps)


g_scheduler = WarmupScheduler(g_optimizer, warmup_steps, g_scheduler_cosine)
d_scheduler = WarmupScheduler(d_optimizer, warmup_steps, d_scheduler_cosine)

frequency = 5
checkpoint = r'weights/experiment_1'
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