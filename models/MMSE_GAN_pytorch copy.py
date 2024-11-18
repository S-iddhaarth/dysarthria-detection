
from torch.autograd import Variable
from torch import nn
import sagan
import torch
import sys
sys.path.append(r'../')
import data_loader
from torch.utils.data import random_split,DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
#parameters
batch_size= 32 

# print("\n\n\n\n\nCuda available:",torch.cuda.is_available(),"\n\n\n\n\n")


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x



adversarial_loss = nn.BCELoss()
mmse_loss = nn.MSELoss()

# Initialization
Gnet = sagan.Generator()
Dnet = sagan.Discriminator()
Gnet.to('cuda')
Dnet.to('cuda')
# Optimizers
optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=0.0001)
# Number of total epochs and warmup steps
epochs = 100
warmup_steps = 10  # Number of steps for warmup

# Initialize the CosineAnnealingLR scheduler
scheduler_G = CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=0)  # T_max is the number of epochs
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=0)

# Function for exponential learning rate warmup
def warmup_lr(optimizer, epoch, warmup_steps=10, initial_lr=0.0001):
    if epoch < warmup_steps:
        lr_scale = (epoch + 1) / warmup_steps  # Linear increase
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr * lr_scale
    else:
        # After warmup, continue with the Cosine Annealing
        scheduler_G.step()  # Apply the cosine annealing scheduler for the generator
        scheduler_D.step()  # Apply the cosine annealing scheduler for the discriminator
root = '../data/UASPEECH'
img_dir = 'feature_extracted'
img_type = "resampling"
annotation = "AlignedResampledMFCC.csv"


dataset = data_loader.SpeechPairLoaderPreprocessed(root, img_dir, img_type, annotation)

# Calculate the number of samples for train and validation
total = len(dataset)
train_size = int(0.8 * total)  # 80% for training
valid_size = total - train_size  # 20% for validation

# Split the dataset into train and validation sets
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# Define the DataLoader for training
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=7,
    persistent_workers=True,  # Keep workers alive between epochs
    pin_memory=True,
    prefetch_factor=4,
    shuffle=True  # Shuffle the training data
)

# Define the DataLoader for validation
valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    num_workers=7,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    shuffle=False  # No need to shuffle validation data
)



# Training Function
def training(data_loader, n_epochs):
    Gnet.train()
    Dnet.train()
    
    for en, (a, b) in enumerate(train_loader):
        a = Variable(a.squeeze(0)).type(torch.cuda.FloatTensor)
        b = Variable(b.squeeze(0)).type(torch.cuda.FloatTensor)

        valid = Variable(torch.Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).type(torch.cuda.FloatTensor)
        fake = Variable(torch.Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).type(torch.cuda.FloatTensor)
        warmup_lr(optimizer_G, epoch, warmup_steps=warmup_steps, initial_lr=0.0001)
        warmup_lr(optimizer_D, epoch, warmup_steps=warmup_steps, initial_lr=0.0001)
        
        optimizer_G.zero_grad()
        Gout = Gnet(a)
        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)*10
        
        G_loss.backward()
        optimizer_G.step()
        
        #G_running_loss = 0
        #G_running_loss += G_loss.item()

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        D_loss.backward()
        optimizer_D.step()
        # D_loss = 0

        #D_running_loss = 0
        #D_running_loss += D_loss.item()
        
        print ("[Epoch: %d] [Iter: %d/%d] [D loss: %f] [G loss: %f]" % (n_epochs, en, len(data_loader), D_loss, G_loss.cpu().data.numpy()))
    


def validating(data_loader):
    Gnet.eval()
    Dnet.eval()
    Grunning_loss = 0
    Drunning_loss = 0

    valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
    
    for en, (a, b) in enumerate(valid_loader):
        a = Variable(a.squeeze(0)).type(torch.cuda.FloatTensor)
        b = Variable(b.squeeze(0)).type(torch.cuda.FloatTensor)
        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).type(torch.cuda.FloatTensor)
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).type(torch.cuda.FloatTensor)

        # optimizer_G.zero_grad()
        Gout = Gnet(a)
        
        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)

        Grunning_loss += G_loss.item()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        Drunning_loss += D_loss.item()
        
    return Drunning_loss/(en+1),Grunning_loss/(en+1)
    
    
isTrain = True

if isTrain:
    epoch = 100
    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_loader, ep+1)
        # if (ep+1)%5==0:
            # torch.save(Gnet, join(mainfolder,"gen_g_{}_d_{}_Ep_{}.pth".format(1,1,ep+1)))
            # torch.save(Dnet, join(mainfolder,"dis_g_{}_d_{}_Ep_{}.pth".format(1,1,ep+1)))
        # dl,gl = validating(valid_loader)
        # print("D_loss: " + str(dl) + " G_loss: " + str(gl))
        # dl_arr.append(dl)
        # gl_arr.append(gl)
        # if ep == 0:
        #     gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Generator'))
        #     dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Discriminator'))
        # else:
        #     viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
        #     viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

            
    # savemat(mainfolder+"/"+str('discriminator_loss.mat'),  mdict={'foo': dl_arr})
    # savemat(mainfolder+"/"+str('generator_loss.mat'),  mdict={'foo': dl_arr})

    # plt.figure(1)
    # plt.plot(dl_arr)
    # plt.savefig(mainfolder+'/discriminator_loss.png')
    # plt.figure(2)
    # plt.plot(gl_arr)
    # plt.savefig(mainfolder+'/generator_loss.png')

else:
    print("Testing")
    save_folder = "/media/"+ str(f_arg) +"/"+ str(s_arg) +"_mask_mmse/"

    test_folder_path = "/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/data/"+ str(s_arg) +"_batches/testing_batches/"
    
    n = len(listdir(test_folder_path))
    Gnet = torch.load(join(mainfolder,"gen_g_1_d_1_Ep_100.pth"), map_location='cpu')
    for i in range(n):
        d = loadmat(join(test_folder_path, "Test_Batch_{}.mat".format(str(i))))
        a = torch.from_numpy(d['Feat'])
        a = Variable(a.squeeze(0).type('torch.FloatTensor'))
        Gout = Gnet(a)
        # np.save(join(save_folder,'Test_Batch_{}.npy'.format(str(i))), Gout.cpu().data.numpy())
        savemat(join(save_folder,'File_{}.mat'.format(str(i))),  mdict={'foo': Gout.cpu().data.numpy()})
