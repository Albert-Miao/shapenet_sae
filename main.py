import torch

import numpy as np
import math

from models import NaivewSAENet, PointNetwSAE, PCEncoderNet, PCDecoderNet, SAENet, PCClassiferNet, PCHDecoderNet
from model_pipelines import trainNet, trainSAE, evalNet, getVis, trainAEEncoder, trainAESAE, trainAEClassifier, getAEVis
from datasets import get_dataloaders

from options import AEOptions

def train(opt):
    torch.cuda.set_device(opt.gpu)
    print(torch.cuda.get_device_name(0))
    
    trainloader, visloader, testloader = get_dataloaders(opt)
   
    if opt.model == 'pointnet':
        net = PointNetwSAE(opt).cuda()
        if opt.load_model_for_sae:
            net.load_state_dict(torch.load("pointnet_03_19_25_sae1_dead_k_256.pth", weights_only=True))
    else:
        net = NaivewSAENet(opt).cuda()
        if opt.load_model_for_sae:
            net.load_state_dict(torch.load("04_16_25_hd_128_codebook_512_no_batchk.pth", weights_only=True))
    
    if not opt.load_model_for_sae:
        trainNet(trainloader, testloader, net, opt)

    if not opt.visualize:
        trainSAE(trainloader, testloader, net, opt)
        
    if opt.visualize:
        getVis(visloader, net, opt)
        
    evalNet(trainloader, testloader, net, opt)
    
def trainAE(opt):
    torch.cuda.set_device(opt.gpu)
    print(torch.cuda.get_device_name(0))
    
    trainloader, visloader, testloader = get_dataloaders(opt)
    
    encoder = PCEncoderNet(opt).cuda()
    if opt.load_encoder:
        encoder.load_state_dict(torch.load(opt.encoder_load_path))
    
    if not opt.h_decoder:
        decoder = PCDecoderNet(opt).cuda()
        if opt.load_decoder:
            decoder.load_state_dict(torch.load(opt.decoder_load_path))
    else:
        decoder = PCHDecoderNet(opt).cuda()
        if opt.load_decoder:
            decoder.load_state_dict(torch.load(opt.decoder_load_path))
    
    sae = SAENet(opt).cuda()
    if opt.load_sae:
        sae.load_state_dict(torch.load(opt.sae_load_path))
 
    classifier = PCClassiferNet(opt).cuda()
    if opt.load_classifier:
        classifier.load_state_dict(torch.load(opt.classifier_load_path))
        
    
    if opt.train_encoder:
        trainAEEncoder(trainloader, encoder, decoder, opt)
    
    encoder.eval()
    
    if opt.train_sae:
        trainAESAE(trainloader, encoder, sae, opt)
    
    if opt.train_classifier:
        trainAEClassifier(trainloader, testloader, encoder, classifier, opt)
    
    if opt.visualize:
        getAEVis(visloader, encoder, decoder, sae, classifier, opt)
    
    
def main():
    options = AEOptions()
    opt = options.parse()
    
    # train(opt)
    trainAE(opt)
    
if __name__ == "__main__":
    main()