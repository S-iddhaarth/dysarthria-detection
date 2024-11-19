import wandb
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from collections import defaultdict
import math
from torch.amp import autocast, GradScaler
from data_loader import SpeechPairLoaderPreprocessed
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerGenerator(nn.Module):
    def __init__(self, input_dim=80, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Length regulator
        self.length_regulator = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_dim, input_dim, 3, padding=1)
        )
    
    def forward(self, src, tgt):
        # Adjust sequence length
        src = src.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        tgt = tgt.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        
        # Input embedding
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        # Generate target sequence length mask
        tgt_len = tgt.size(1)
        src = F.interpolate(src.transpose(1, 2), size=tgt_len, mode='linear', align_corners=False)
        src = src.transpose(1, 2)
        
        # Transformer encoding
        memory = self.transformer_encoder(src)
        
        # Create target mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(src.device)
        
        # Decoder input preparation
        tgt = self.input_projection(tgt)
        tgt = self.pos_encoder(tgt)
        
        # Transformer decoding
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.output_projection(output)
        
        # Transpose back to [B, C, T]
        output = output.transpose(1, 2)
        
        # Fine-tune the temporal resolution
        output = self.length_regulator(output)
        
        return output

class Discriminator(nn.Module):
    def __init__(self, input_dim=80, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.output_layer(x)

class MelSpectrogramTransformer:
    def __init__(self, device='cuda', learning_rate=0.0001):
        self.device = device
        self.generator = TransformerGenerator().to(device)
        self.discriminator = Discriminator().to(device)
        
        self.criterion_gan = nn.MSELoss()
        self.criterion_recon = nn.L1Loss()
        
        self.optimizer_G = torch.optim.AdamW(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        self.optimizer_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.scaler_G = GradScaler('cuda')
        self.scaler_D = GradScaler('cuda')
    
    def cosine_similarity(self, x1, x2):
        return F.cosine_similarity(x1.flatten(1), x2.flatten(1), dim=1).mean()
    
    def train_step(self, real_A, real_B):
        # Train Discriminator
        with autocast('cuda'):
            fake_B = self.generator(real_A, real_B)
            
            # Real loss
            pred_real = self.discriminator(real_B)
            loss_D_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))
            
            # Fake loss
            pred_fake = self.discriminator(fake_B.detach())
            loss_D_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        self.optimizer_D.zero_grad()
        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.step(self.optimizer_D)
        self.scaler_D.update()
        
        # Train Generator
        with autocast('cuda'):
            pred_fake = self.discriminator(fake_B)
            loss_G_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
            
            # Reconstruction loss
            loss_G_recon = self.criterion_recon(fake_B, real_B) * 10.0
            
            # Cosine similarity loss
            cos_sim = self.cosine_similarity(fake_B, real_B)
            loss_G_cos = (1 - cos_sim) * 5.0
            
            loss_G = loss_G_gan + loss_G_recon + loss_G_cos
        
        self.optimizer_G.zero_grad()
        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()
        
        return {
            'loss_D': loss_D.item(),
            'loss_G': loss_G.item(),
            'loss_G_gan': loss_G_gan.item(),
            'loss_G_recon': loss_G_recon.item(),
            'loss_G_cos': loss_G_cos.item(),
            'cosine_similarity': cos_sim.item()
        }
def create_dataloaders(batch_size=32):
    dataset = SpeechPairLoaderPreprocessed(
        root='data/UASPEECH',
        img_dir='mfcc_low',
        img_type="reflect_interpolate",
        annotation="output_low_reduced.csv",
        select=1,
    )
    
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=7,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=7,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=False
    )
    
    return train_loader, valid_loader

def validate(model, valid_loader, device):
    model.generator.eval()
    model.discriminator.eval()
    
    val_metrics = defaultdict(float)
    
    with torch.no_grad():
        for dys, cont in valid_loader:
            dys = dys.to(device)
            cont = cont.to(device)
            
            with autocast('cuda'):
                fake_B = model.generator(dys, cont)
                
                # Discriminator outputs
                pred_real = model.discriminator(cont)
                pred_fake = model.discriminator(fake_B)
                
                # Calculate losses
                loss_D_real = model.criterion_gan(pred_real, torch.ones_like(pred_real))
                loss_D_fake = model.criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                
                # Generator losses
                loss_G_gan = model.criterion_gan(pred_fake, torch.ones_like(pred_fake))
                loss_G_recon = model.criterion_recon(fake_B, cont) * 10.0
                cos_sim = model.cosine_similarity(fake_B, cont)
                loss_G_cos = (1 - cos_sim) * 5.0
                loss_G = loss_G_gan + loss_G_recon + loss_G_cos
                
                # Accumulate metrics
                val_metrics['val_loss_D'] += loss_D.item()
                val_metrics['val_loss_G'] += loss_G.item()
                val_metrics['val_loss_G_gan'] += loss_G_gan.item()
                val_metrics['val_loss_G_recon'] += loss_G_recon.item()
                val_metrics['val_loss_G_cos'] += loss_G_cos.item()
                val_metrics['val_cosine_similarity'] += cos_sim.item()
    
    # Calculate averages
    for key in val_metrics:
        val_metrics[key] /= len(valid_loader)
    
    model.generator.train()
    model.discriminator.train()
    
    return val_metrics

def save_checkpoint(model, epoch, metrics, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'optimizer_G_state_dict': model.optimizer_G.state_dict(),
        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))

def train(model, train_loader, valid_loader, num_epochs, device):
    # Initialize wandb
    wandb.init(
        job_type="naive_run",
        project="The last dance",
        entity="PaneerShawarma",
    )
    
    # Log model architecture
    wandb.watch(model.generator, log='all')
    wandb.watch(model.discriminator, log='all')
    
    for epoch in range(num_epochs):
        epoch_metrics = defaultdict(float)
        
        # Training loop
        for i, (dys, cont) in enumerate(train_loader):
            dys = dys.to(device)
            cont = cont.to(device)
            
            metrics = model.train_step(dys, cont)
            
            for key, value in metrics.items():
                epoch_metrics[key] += value
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] "
                      f"Batch [{i}/{len(train_loader)}] "
                      f"Loss_D: {metrics['loss_D']:.4f} "
                      f"Loss_G: {metrics['loss_G']:.4f} "
                      f"Cos_Sim: {metrics['cosine_similarity']:.4f}")
        
        # Calculate training averages
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)
        
        # Validation
        val_metrics = validate(model, valid_loader, device)
        
        # Combine metrics for logging
        log_metrics = {
            # Training metrics
            'train/loss_D': epoch_metrics['loss_D'],
            'train/loss_G': epoch_metrics['loss_G'],
            'train/loss_G_gan': epoch_metrics['loss_G_gan'],
            'train/loss_G_recon': epoch_metrics['loss_G_recon'],
            'train/loss_G_cos': epoch_metrics['loss_G_cos'],
            'train/cosine_similarity': epoch_metrics['cosine_similarity'],
            # Validation metrics
            'val/loss_D': val_metrics['val_loss_D'],
            'val/loss_G': val_metrics['val_loss_G'],
            'val/loss_G_gan': val_metrics['val_loss_G_gan'],
            'val/loss_G_recon': val_metrics['val_loss_G_recon'],
            'val/loss_G_cos': val_metrics['val_loss_G_cos'],
            'val/cosine_similarity': val_metrics['val_cosine_similarity'],
            'epoch': epoch
        }
        
        # Log to wandb
        wandb.log(log_metrics)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print("Training Metrics:")
        for key, value in epoch_metrics.items():
            print(f"Average {key}: {value:.4f}")
        print("\nValidation Metrics:")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")
        print()
        
        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, epoch, {**epoch_metrics, **val_metrics})
    
    wandb.finish()

# Main execution
if __name__ == "__main__":
    # Create model and dataloaders
    model = MelSpectrogramTransformer(device='cuda')
    train_loader, valid_loader = create_dataloaders(batch_size=32)
    
    # Train model
    train(model, train_loader, valid_loader, num_epochs=100, device='cuda')