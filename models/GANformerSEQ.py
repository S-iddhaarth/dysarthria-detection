import torch
import torch.nn as nn
import math

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        position = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        position = position.unsqueeze(0).expand(x.size(0), x.size(1))
        return x + self.pos_embedding(position)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = torch.mean(x, dim=1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.sigmoid(z).unsqueeze(1)
        return x * z.unsqueeze(-1)

class Generator(nn.Module):
    def __init__(self, input_channels=80, output_channel=80, 
                 input_seq_len=250, output_seq_len=125,
                 d_model=250, nhead=10, num_layers=12, 
                 dim_feedforward=2048, dropout=0.2):
        super().__init__()

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        # Initial projection from input channels to d_model
        self.input_proj = nn.Linear(input_channels, d_model)

        # Learnable positional encoding
        self.pos_encoder = LearnablePositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Sequence length reduction using convolution
        self.seq_reduction = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, output_channel)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Squeeze-and-Excitation
        self.se_block = SEBlock(d_model)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, 80, 250]
        x = x.transpose(1, 2)  # [batch_size, 250, 80]

        # Project to transformer dimensions
        x = self.input_proj(x)  # [batch_size, 250, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Apply squeeze-and-excitation
        x = self.se_block(x)

        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Reduce sequence length using convolution
        x = x.transpose(1, 2)  # [batch_size, d_model, 250]
        x = self.seq_reduction(x)  # [batch_size, d_model, 125]
        x = x.transpose(1, 2)  # [batch_size, 125, d_model]

        # Project to output channels
        x = self.output_proj(x)  # [batch_size, 125, 80]

        # Final shape adjustment
        x = x.transpose(1, 2)  # [batch_size, 80, 125]
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=80, 
                 input_seq_len=250, output_seq_len=125,
                 d_model=512, nhead=16, num_layers=12, 
                 dim_feedforward=2048, dropout=0.2):
        super().__init__()

        # Sequence matching using convolution
        self.seq_match = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        )

        # Initial projection (input + output concatenated)
        self.input_proj = nn.Linear(input_channels * 2, d_model)

        # Learnable positional encoding
        self.pos_encoder = LearnablePositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.relu = nn.LeakyReLU(0.2)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Squeeze-and-Excitation
        self.se_block = SEBlock(d_model)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        # x shape: [batch_size, 80, 250]
        # y shape: [batch_size, 80, 125]
        
        # Match sequence lengths using convolution
        x = self.seq_match(x)  # [batch_size, 80, 125]

        # Reshape for transformer
        x = x.transpose(1, 2)  # [batch_size, 125, 80]
        y = y.transpose(1, 2)  # [batch_size, 125, 80]

        # Concatenate along channel dimension
        xy = torch.cat([x, y], dim=-1)  # [batch_size, 125, 160]

        # Project to transformer dimensions
        xy = self.input_proj(xy)

        # Add positional encoding
        xy = self.pos_encoder(xy)

        # Pass through transformer
        xy = self.transformer_encoder(xy)

        # Apply squeeze-and-excitation
        xy = self.se_block(xy)

        # Apply layer normalization and dropout
        xy = self.layer_norm(xy)
        xy = self.dropout(xy)

        # Global average pooling
        xy = torch.mean(xy, dim=1)

        # Final classification
        xy = self.relu(self.fc1(xy))
        xy = self.fc2(xy)
        return xy