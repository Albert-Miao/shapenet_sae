import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AEwSAENet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.sae_dim = opt.codebook_size
        self.batch_size = opt.batch_size
        self.l1_lambda = opt.l1_lambda
        
        self.criterion = nn.MSELoss()
        self.l1criterion = nn.L1Loss()
        
        self.enc1 = nn.Conv3d(1, 6, 5, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.enc2 = nn.Conv3d(6, 10, 5)
        
        self.fc1 = nn.Linear(1250, 800)
        self.fc2 = nn.Linear(800, self.hidden_rep_dim)
        self.fc3 = nn.Linear(self.hidden_rep_dim, 800)
        self.fc4 = nn.Linear(800, 1250)
                
        self.upsample = nn.Upsample(scale_factor=2)
        self.dec1 = nn.ConvTranspose3d(10, 6, 5)
        self.dec2 = nn.ConvTranspose3d(6, 1, 5, padding=1)
        
        self.sae1 = nn.Linear(self.hidden_rep_dim, self.sae_dim)
        self.sae2 = nn.Linear(self.sae_dim, self.hidden_rep_dim)
        
        self.sae1.requires_grad_(False)
        self.sae2.requires_grad_(False)

        with torch.no_grad():
            self.sae1.weight[:, :].copy_(self.sae2.weight.T)
            self.sae1.bias[:].copy_(torch.zeros(self.sae_dim).cuda())
            self.sae2.bias[:].copy_(torch.zeros(self.hidden_rep_dim).cuda())
        
        self.model_optimizer = optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.momentum)
        self.sae_optimzier = optim.Adam(self.parameters(), lr=self.opt.lr)
        
        self.optimizer = self.model_optimizer
        
    def eval(self):
        super().eval()
        self.change_stage(0)
        
    def forward(self, x):
        x = self.pool(F.relu(self.enc1(x)))
        x = self.pool(F.relu(self.enc2(x)))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.stage % 2 != 0:
            _x = x - self.sae2.bias
            f = F.relu(self.sae1(_x))
            _x = self.sae2(f)
            
            return x, _x, f
        
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.reshape(x, (self.batch_size, 10, 5, 5, 5))
        x = self.upsample(x)
        x = F.relu(self.dec1(x))
        x = self.upsample(x)
        x = self.dec2(x)
        
        return x
        
    def change_stage(self, stage):
        if stage == 1:
            self.stage = 1
            
            self.sae1.requires_grad_(True)
            self.sae2.requires_grad_(True)
            
            self.enc1.requires_grad_(False)
            self.enc2.requires_grad_(False)
            self.fc1.requires_grad_(False)
            self.fc2.requires_grad_(False)
            
            self.optmizer = self.sae_optimzier
        else:
            self.stage = 0
            
            self.sae1.requires_grad_(False)
            self.sae2.requires_grad_(False)
            
            self.enc1.requires_grad_(True)
            self.enc2.requires_grad_(True)
            self.fc1.requires_grad_(True)
            self.fc2.requires_grad_(True)
            
            self.optimizer = self.model_optimizer

                
        
            
                