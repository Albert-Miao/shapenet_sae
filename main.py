import torch

import numpy as np
import math

from models import AEwSAENet
from model_pipelines import trainNet, evalNet

from options import AEOptions

def train(opt):
    torch.cuda.set_device(opt.gpu)
    print(torch.cuda.get_device_name(0))
    
    net = AEwSAENet(opt).cuda()
    
    trainloader, testloader = generate_data_loaders(opt)
    trainNet(trainloader, testloader, net, opt)
    evalNet(trainloader, testloader, net, opt)
    
    
def main():
    options = AEOptions()
    opt = options.parse()
    
    train(opt)
    
if __name__ == "__main__":
    main()