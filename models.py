import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.autograd import Variable

from chamferdist import ChamferDistance
from utils import JumpReLU

class PCEncoderNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.batch_size = opt.batch_size
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, self.hidden_rep_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(self.hidden_rep_dim)
        
        self.bn6 = nn.BatchNorm1d(self.hidden_rep_dim)
        # self.bn6 = nn.LayerNorm([self.hidden_rep_dim])
        
        self.fc = nn.Linear(self.hidden_rep_dim, self.hidden_rep_dim)
         
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
            
        x = F.tanh(self.fc(torch.flatten(x, 1)))
        # x = self.bn6(F.relu(self.fc(torch.flatten(x, 1))))
        # x = F.relu(self.fc(torch.flatten(x, 1)))
        
        return x
    
class SAENet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.batch_size = opt.batch_size
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.sae_dim = opt.codebook_size
        self.l1_lambda = opt.l1_lambda
        
        self.dead_features = torch.zeros((opt.codebook_size))
        
        self.recon_criterion = nn.MSELoss()
        self.l1criterion = nn.L1Loss()
        
        self.sae1 = nn.Linear(self.hidden_rep_dim, self.sae_dim)
        self.sae2 = nn.Linear(self.sae_dim, self.hidden_rep_dim)

        with torch.no_grad():
            self.sae1.weight[:, :].copy_(self.sae2.weight.T)
            self.sae1.bias[:].copy_(torch.zeros(self.sae_dim).cuda())
            self.sae2.bias[:].copy_(torch.zeros(self.hidden_rep_dim).cuda())
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.opt.sae_lr)        
        
    def forward(self, x):
        _x = x - self.sae2.bias
        f = F.relu(self.sae1(_x))
        
        dead_f = torch.zeros_like(f)
        if self.opt.batch_topk:
            f = f.flatten()
            topk, inds = f.topk(self.opt.k * x.size(0))
            
            dead_features_inds = self.dead_features.unsqueeze(0).repeat(x.size(0), 1).flatten() >= 5
            dead_f = torch.zeros_like(f)
            dead_f[dead_features_inds] = f[dead_features_inds]
            dead_topk, dead_inds = dead_f.topk(min((dead_features_inds != 0).sum(), self.opt.dead_k * x.size(0)))
            
            f = torch.zeros_like(f)
            dead_f = torch.zeros_like(f)
            
            f[inds] = topk
            f = f.unflatten(0, (x.size(0), -1))
            
            dead_f[dead_inds] = dead_topk
            dead_f = dead_f.unflatten(0, (x.size(0), -1))
            
        _x = self.sae2(f)
        dead_x = self.sae2(dead_f)
        
        return _x, f, dead_x
    
class JumpSAENet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.batch_size = opt.batch_size
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.sae_dim = opt.codebook_size
        self.l1_lambda = opt.l1_lambda
        self.t = opt.t
        
        self.dead_features = torch.zeros((opt.codebook_size))
        
        self.recon_criterion = nn.MSELoss()
        self.l1criterion = nn.L1Loss()
        
        self.sae1 = nn.Linear(self.hidden_rep_dim, self.sae_dim)
        self.sae2 = nn.Linear(self.sae_dim, self.hidden_rep_dim)
        self.jumpAct = JumpReLU(self.t)

        with torch.no_grad():
            self.sae1.weight[:, :].copy_(self.sae2.weight.T)
            self.sae1.bias[:].copy_(torch.zeros(self.sae_dim).cuda())
            self.sae2.bias[:].copy_(torch.zeros(self.hidden_rep_dim).cuda())
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.opt.sae_lr)        
        
    def forward(self, x):
        _x = x - self.sae2.bias
        f = self.jumpAct(self.sae1(_x))
        
        dead_f = torch.zeros_like(f)
        if self.opt.batch_topk:
            f = f.flatten()
            topk, inds = f.topk(self.opt.k * x.size(0))
            
            dead_features_inds = self.dead_features.unsqueeze(0).repeat(x.size(0), 1).flatten() >= 5
            dead_f = torch.zeros_like(f)
            dead_f[dead_features_inds] = f[dead_features_inds]
            dead_topk, dead_inds = dead_f.topk(min((dead_features_inds != 0).sum(), self.opt.dead_k * x.size(0)))
            
            f = torch.zeros_like(f)
            dead_f = torch.zeros_like(f)
            
            f[inds] = topk
            f = f.unflatten(0, (x.size(0), -1))
            
            dead_f[dead_inds] = dead_topk
            dead_f = dead_f.unflatten(0, (x.size(0), -1))
            
        _x = self.sae2(f)
        dead_x = self.sae2(dead_f)
        
        return _x, f, dead_x
    
class PCDecoderNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.batch_size = opt.batch_size
        self.point_size = opt.num_points
        
        self.dec1 = nn.Linear(self.hidden_rep_dim, 1024)
        self.dec2 = nn.Linear(1024, 1024)
        self.dec3 = nn.Linear(1024, self.point_size * 3)
        
        self.criterion = ChamferDistance()
        # self.optimizer = optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.momentum)
        self.optimizer = optim.Adam(self.parameters(), lr=self.opt.lr)        
       
    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        
        return x.view(-1, self.point_size, 3)

class PCHDecoderNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.batch_size = opt.batch_size
        self.point_size = opt.num_points
        
        self.fc1 = nn.Linear(self.hidden_rep_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.feat = nn.Linear(512, 64 * 64)
        self.xyz = nn.Linear(512, 64 * 3)
        self.bn3 = nn.BatchNorm1d(64 * 64)
        self.bn4 = nn.BatchNorm1d(64 * 3)
        
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, self.point_size // 64 * 3, 1)
        
        self.criterion = ChamferDistance()
        self.optimizer = optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.momentum)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        x_feat = F.relu(self.bn3(self.feat(x)))
        xyz_base = F.relu(self.bn4(self.xyz(x)))
        
        x_feat = x_feat.view(-1, 64, 64)
        x_base = xyz_base.view(-1, 64, 3).unsqueeze(2).repeat(1, 1, self.point_size // 64, 1)
        
        x = self.conv1(x_feat)
        x = self.conv2(x_feat)
        x = x.swapaxes(1,2).view(-1, 64, self.point_size // 64, 3)
        x = x + x_base
        
        return x.reshape(-1, self.point_size, 3)
    
class PCClassiferNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.batch_size = opt.batch_size
        self.point_size = opt.num_points
        
        self.fc1 = nn.Linear(opt.hidden_rep_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 55)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
  

class NaivewSAENet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        
        self.stage = 0
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.sae_dim = opt.codebook_size
        self.batch_size = opt.batch_size
        self.l1_lambda = opt.l1_lambda
        
        self.dead_features = torch.zeros((opt.codebook_size))
        
        self.criterion = nn.CrossEntropyLoss()
        self.recon_criterion = nn.MSELoss()
        self.l1criterion = nn.L1Loss()
        
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, self.hidden_rep_dim)
        self.fc3 = nn.Linear(self.hidden_rep_dim, 64)
        self.fc4 = nn.Linear(64, 55)
        
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
            
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.stage % 2 != 0:
            return self.run_sae(self.sae1, self.sae2, x)
        
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
        
    def run_sae(self, sae1, sae2, x):
        _x = x - sae2.bias
        f = F.relu(sae1(_x))
        
        dead_f = torch.zeros_like(f)
        if self.opt.batch_topk:
            f = f.flatten()
            topk, inds = f.topk(self.opt.k * x.size(0))
            
            dead_features_inds = self.dead_features.unsqueeze(0).repeat(x.size(0), 1).flatten() >= 5
            dead_f = torch.zeros_like(f)
            dead_f[dead_features_inds] = f[dead_features_inds]
            dead_topk, dead_inds = dead_f.topk(min((dead_features_inds != 0).sum(), self.opt.dead_k * x.size(0)))
            
            f = torch.zeros_like(f)
            dead_f = torch.zeros_like(f)
            
            f[inds] = topk
            f = f.unflatten(0, (x.size(0), -1))
            
            dead_f[dead_inds] = dead_topk
            dead_f = dead_f.unflatten(0, (x.size(0), -1))
            
        _x = sae2(f)
        dead_x = sae2(dead_f)
        
        return x, _x, f, dead_x

    
    def change_stage(self, stage):
        if stage == 1:
            self.stage = 1
            
            self.sae1.requires_grad_(True)
            self.sae2.requires_grad_(True)
            
            self.conv1.requires_grad_(False)
            self.conv2.requires_grad_(False)
            self.conv3.requires_grad_(False)
            self.conv4.requires_grad_(False)
            self.bn1.requires_grad_(False)
            self.bn2.requires_grad_(False)
            self.bn3.requires_grad_(False)
            self.bn4.requires_grad_(False)

            self.fc1.requires_grad_(False)
            self.fc2.requires_grad_(False)
            
            self.optmizer = self.sae_optimzier
        else:
            self.stage = 0
            
            self.sae1.requires_grad_(False)
            self.sae2.requires_grad_(False)
            
            self.bn1.requires_grad_(True)
            self.bn2.requires_grad_(True)
            self.bn3.requires_grad_(True)
            self.bn4.requires_grad_(True)

            self.fc1.requires_grad_(True)
            self.fc2.requires_grad_(True)           
            
            self.optimizer = self.model_optimizer

                
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
    
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x    
    
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
            
            
class PointNetwSAE(nn.Module):
    def __init__(self, opt, k=55):
        super(PointNetwSAE, self).__init__()
        
        self.opt = opt
        self.stage = 0
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.sae_dim = opt.codebook_size
        
        channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.criterion = get_loss()
        self.recon_criterion = nn.MSELoss()
        self.l1criterion = nn.L1Loss()
        
        self.sae1 = nn.Linear(512, self.sae_dim)
        self.sae2 = nn.Linear(self.sae_dim, 512)
        
        self.sae3 = nn.Linear(256, self.sae_dim)
        self.sae4 = nn.Linear(self.sae_dim, 256)
        
        self.sae1.requires_grad_(False)
        self.sae2.requires_grad_(False)
        self.sae3.requires_grad_(False)
        self.sae4.requires_grad_(False)
        
        self.dead_features = torch.zeros((opt.codebook_size)).cuda()
        
        with torch.no_grad():
            self.sae1.weight[:, :].copy_(self.sae2.weight.T)
            self.sae1.bias[:].copy_(torch.zeros(self.sae_dim).cuda())
            self.sae2.bias[:].copy_(torch.zeros(512).cuda())
            
            self.sae3.weight[:, :].copy_(self.sae4.weight.T)
            self.sae3.bias[:].copy_(torch.zeros(self.sae_dim).cuda())
            self.sae4.bias[:].copy_(torch.zeros(256).cuda())

            
        self.model_optimizer = optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.momentum)
        self.sae_optimzier = optim.Adam(self.parameters(), lr=self.opt.lr)
        
        self.optimizer = self.model_optimizer

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        
        if self.stage % 2 != 0 and self.opt.pointnet_sae_level == 1:
            return self.run_sae(self.sae1, self.sae2, x)
        
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        
        if self.stage % 2 != 0 and self.opt.pointnet_sae_level == 2:
            return self.run_sae(self.sae3, self.sae4, x)
                
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat
    
    def run_sae(self, sae1, sae2, x):
        _x = x - sae2.bias
        f = F.relu(sae1(_x))
        
        dead_f = torch.zeros_like(f)
        if self.opt.batch_topk:
            f = f.flatten()
            topk, inds = f.topk(self.opt.k * x.size(0))
            
            dead_features_inds = self.dead_features.unsqueeze(0).repeat(x.size(0), 1).flatten() >= 5
            dead_f = torch.zeros_like(f)
            dead_f[dead_features_inds] = f[dead_features_inds]
            dead_topk, dead_inds = dead_f.topk(min((dead_features_inds != 0).sum(), self.opt.dead_k * x.size(0)))
            
            f = torch.zeros_like(f)
            dead_f = torch.zeros_like(f)
            
            f[inds] = topk
            f = f.unflatten(0, (x.size(0), -1))
            
            dead_f[dead_inds] = dead_topk
            dead_f = dead_f.unflatten(0, (x.size(0), -1))

            
        _x = sae2(f)
        dead_x = sae2(dead_f)
        
        return x, _x, f, dead_x

    def eval(self):
        super().eval()
        # self.change_stage(0)
    
    def change_stage(self, stage):
        if stage == 1:
            self.stage = 1
            
            self.sae1.requires_grad_(True)
            self.sae2.requires_grad_(True)
            self.sae3.requires_grad_(True)
            self.sae4.requires_grad_(True)           
            
            self.feat.requires_grad_(False)
            self.fc1.requires_grad_(False)
            self.fc2.requires_grad_(False)
            self.fc3.requires_grad_(False)
            self.bn1.requires_grad_(False)
            self.bn2.requires_grad_(False)
            
            self.optmizer = self.sae_optimzier
        else:
            self.stage = 0
            
            self.sae1.requires_grad_(False)
            self.sae2.requires_grad_(False)        
            self.sae3.requires_grad_(False)
            self.sae4.requires_grad_(False)        
            
            self.feat.requires_grad_(True)
            self.fc1.requires_grad_(True)
            self.fc2.requires_grad_(True)
            self.fc3.requires_grad_(True)
            self.bn1.requires_grad_(True)
            self.bn2.requires_grad_(True)
            
            self.optimizer = self.model_optimizer

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss