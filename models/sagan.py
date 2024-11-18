import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modelsummary import summary
# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1)
        self.attn1 = SelfAttention(256)
        self.down3 = nn.Conv2d(256, 1024, kernel_size=4, stride=2, padding=1)
        self.attn2 = SelfAttention(1024)
        self.up1 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, output_padding=(0, 0))
        self.up2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=(1, 0))
        self.attn3 = SelfAttention(64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(1, 0))
        self.up4 = nn.Conv2d(32,1,kernel_size=1,padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = F.leaky_relu(self.down1(x), 0.2)
        x = F.leaky_relu(self.down2(x), 0.2)
        x = self.attn1(x)
        x = F.leaky_relu(self.down3(x), 0.2)
        x = self.attn2(x)
        x = F.leaky_relu(self.up1(x), 0.2)
        x = F.leaky_relu(self.up2(x), 0.2)
        x = self.attn3(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.tanh(x)
        return x

# Kaiming Initialization
def weights_init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming SelfAttention is defined elsewhere
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.attn1 = SelfAttention(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.attn2 = SelfAttention(256)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        # Linear layer to output a 2-dimensional vector
        self.fc = nn.Linear(31, 1)  # Input size of 31 (from the last conv output) and output size of 2
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.attn1(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.attn2(x)
        x = self.conv4(x)  # Output size: (batch_size, 1, 1, 31)
        
        # Flatten the output from (batch_size, 1, 1, 31) to (batch_size, 31)
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, 31)
        
        # Pass through the fully connected layer
        x = self.fc(x)  # Output size: (batch_size, 2)
        x = F.sigmoid(x)
        return x


# Kaiming Initialization
def weights_init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)