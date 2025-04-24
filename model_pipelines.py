import torch
from models import get_loss

import numpy as np
import open3d as o3d
import cv2

import os

# from pytorch3d.loss import chamfer_distance
from spconv.pytorch.utils import PointToVoxel

def trainNet(trainloader, testloader, net, opt):
    for epoch in range(opt.num_model_epochs):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader):
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            net.optimizer.zero_grad()
            
            if opt.model == 'pointnet':
                outputs, trans_feat = net(inputs)
                loss = net.criterion(outputs, labels, trans_feat)
            else:
                outputs = net(inputs)
                loss = net.criterion(outputs, labels)
            
            running_loss += loss.item()
            
            loss.backward()
            net.optimizer.step()
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / opt.super_batch_size:.3f}')
                
                running_loss = 0.0
                
        if epoch % 8 == 7:
            train_acc, test_acc = evalNet(trainloader, testloader, net, opt)
                
    PATH = opt.model_path
    torch.save(net.state_dict(), PATH)
    
    return net
    
def trainAEEncoder(trainloader, encoder, decoder, opt):
    for epoch in range(opt.num_model_epochs):
        running_loss = 0.0
        running_mmcr_loss = 0.0
        
        for i, data in enumerate(trainloader):
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            encoder.optimizer.zero_grad()
            decoder.optimizer.zero_grad()
            
            latents = encoder(inputs)
            outputs = decoder(latents)
            
            loss = decoder.criterion(outputs, inputs.transpose(1, 2), bidirectional=True, point_reduction="mean")
            
            running_loss += loss.item()
            
            if opt.enc_MMCR_loss > 0:
                S = torch.linalg.svdvals(torch.nn.functional.normalize(latents, dim=1))
                mmcr_loss = - S.mean() * opt.enc_MMCR_loss
                loss += mmcr_loss
                running_mmcr_loss += mmcr_loss.item()
            
            
            loss.backward()
            encoder.optimizer.step()
            decoder.optimizer.step()
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] recon_loss: {running_loss / opt.super_batch_size:.3f}')
                if opt.enc_MMCR_loss > 0:
                    print(f'MMCR loss: {running_mmcr_loss / opt.super_batch_size:.6f}')
                
                running_loss = 0.0
                running_mmcr_loss = 0.0
        
        if epoch % 50 == 49:
            torch.save(encoder.state_dict(), opt.encoder_save_path)
            torch.save(decoder.state_dict(), opt.decoder_save_path)
    
    return encoder, decoder
    
def trainAESAE(trainloader, encoder, sae, opt):
    dead_features = torch.zeros((opt.codebook_size)).cuda()
    for epoch in range(opt.num_sae_epochs):
        running_recon_loss = 0.0
        running_l1_loss = 0.0
        running_dead_recon_loss = 0.0
        running_mmcr_loss = 0.0
        running_kart_loss = 0.0
        
        features_fired = torch.zeros((opt.codebook_size)).cuda()
        for i, data in enumerate(trainloader):
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            sae.optimizer.zero_grad()
            
            latents = encoder(inputs)
            _latents, fs, dead_x = sae(latents)
            
            recon_loss = sae.recon_criterion(latents, _latents)
            dead_recon_loss = sae.recon_criterion((latents - _latents).clone().detach(), dead_x) * opt.dead_lambda
            l1_loss = torch.tanh(opt.c * fs * torch.linalg.vector_norm(sae.sae2.weight, ord=1, dim=0).unsqueeze(0).repeat(opt.batch_size, 1)).sum() * opt.lambda_s
            
            features_fired += (fs != 0).sum(dim=0)
            
            # MMCR paper, negative nuclear norm. Must be independent of magnitude
            if opt.MMCR_loss > 0:
                S = torch.linalg.svdvals(torch.nn.functional.normalize(sae.sae2.weight, dim=0))
                mmcr_loss = - S.mean() * opt.MMCR_loss           
            
            # We note that the norm of encoder weights for features that fire are significantly larger (x5) than those of dead features, preventing them from ever being 'selected' by topk.
            # We thus encourage encoder weight vectors to approach magnitude of one to give features slower to start a boost, and nerf those in the lead. Note that the magnitude selected 
            # to approach doesn't actually matter, since the magitudes of f are only relevant relative to each other for selection, and can be comepensated for in decoder weights.
            if opt.KART_loss > 0:
                kart_loss = torch.sum((sae.sae1.weight.norm(dim=1) - 1) ** 2) * opt.KART_loss
                
            running_recon_loss += recon_loss.item()
            running_dead_recon_loss += dead_recon_loss.item()
            running_l1_loss += l1_loss.item()
                        
            if opt.batch_topk:
                loss = recon_loss + dead_recon_loss
            else:
                loss = recon_loss + l1_loss
            
            if opt.MMCR_loss > 0:
                loss += mmcr_loss
                running_mmcr_loss += mmcr_loss.item()
                
            if opt.KART_loss > 0:
                loss += kart_loss
                running_kart_loss += kart_loss.item()
       
            loss.backward()
            sae.optimizer.step()
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                if opt.batch_topk:
                    print(f'[{epoch + 1}, {i + 1:5d}] recon loss: {running_recon_loss / opt.super_batch_size:.3f} ' + 
                          f'dead recon loss: {running_dead_recon_loss / opt.super_batch_size:.8f}')
                else:
                    print(f'[{epoch + 1}, {i + 1:5d}] recon loss: {running_recon_loss / opt.super_batch_size:.3f} ' + 
                          f'l1 loss: {running_l1_loss / opt.super_batch_size:.3f}')
                    
                if opt.MMCR_loss > 0:
                    print(f'MMCR loss: {running_mmcr_loss / opt.super_batch_size:.3f}')
                
                if opt.KART_loss > 0:
                    print(f'KART loss: {running_kart_loss / opt.super_batch_size:.3f}')
                
                running_recon_loss = 0.0
                running_dead_recon_loss = 0.0
                running_l1_loss = 0.0
                running_mmcr_loss = 0.0
                running_kart_loss = 0.0
                
        avg_features_fired = features_fired.sum() / len(trainloader.dataset)
        # dead_features = (features_fired == 0).nonzero()[:, 0]
        dead_features += features_fired == 0
        dead_features[features_fired != 0] = 0
        num_dead_features = (dead_features >= 5).sum()
        
        print(avg_features_fired)
        print(num_dead_features)
        
        sae.dead_features = dead_features
        
        if epoch % 50 == 49:
            torch.save(sae.state_dict(), opt.sae_save_path)
    
    return sae

def trainAEClassifier(trainloader, testloader, encoder, classifier, opt):
    for epoch in range(opt.num_model_epochs):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader):
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            classifier.optimizer.zero_grad()
            
            latents = encoder(inputs)
            outputs = classifier(latents)
            loss = classifier.criterion(outputs, labels)
            
            running_loss += loss.item()
            
            loss.backward()
            classifier.optimizer.step()
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / opt.super_batch_size:.3f}')
                
                running_loss = 0.0
                
        if epoch % 8 == 7:
            train_acc, test_acc = evalAENet(trainloader, testloader, encoder, classifier, opt)
                
    torch.save(classifier.state_dict(), opt.classifier_save_path)
    
    return classifier
    
    
def trainSAE(trainloader, testloader, net, opt):
    net.change_stage(1)
    dead_features = torch.zeros((opt.codebook_size)).cuda()
    for epoch in range(opt.num_sae_epochs):
        running_recon_loss = 0.0
        running_l1_loss = 0.0
        running_dead_recon_loss = 0.0
        
        features_fired = torch.zeros((opt.codebook_size)).cuda()
        
        for i, data in enumerate(trainloader):
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            
            net.optimizer.zero_grad()
            
            acts, recons, fs, dead_acts = net(inputs)
            recon_loss = net.recon_criterion(acts, recons)
            dead_recon_loss = net.recon_criterion(dead_acts, (acts - recons).clone().detach()) * opt.dead_lambda
            # l1_loss = net.l1criterion(fs, torch.zeros_like(fs)) * opt.l1_lambda
            l1_loss = torch.sum(torch.sum(torch.abs(fs), dim=0) * torch.linalg.vector_norm(net.sae2.weight, ord=2, dim=0))
            
            features_fired += (fs != 0).sum(dim=0)
            
            running_recon_loss += recon_loss.item()
            running_dead_recon_loss += dead_recon_loss.item()
            running_l1_loss += l1_loss.item()
            
            if opt.batch_topk:
                loss = recon_loss + dead_recon_loss
            else:
                loss = recon_loss + l1_loss
            
            loss.backward()
            net.optimizer.step()
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                if opt.batch_topk:
                    print(f'[{epoch + 1}, {i + 1:5d}] recon loss: {running_recon_loss / opt.super_batch_size:.3f} ' + 
                          f'dead recon loss: {running_dead_recon_loss / opt.super_batch_size:.8f}')
                else:
                    print(f'[{epoch + 1}, {i + 1:5d}] recon loss: {running_recon_loss / opt.super_batch_size:.3f} ' + 
                          f'l1 loss: {running_l1_loss / opt.super_batch_size:.3f}')
                
                running_recon_loss = 0.0
                running_dead_recon_loss = 0.0
                running_l1_loss = 0.0
                
        avg_features_fired = features_fired.sum() / len(trainloader.dataset)
        # dead_features = (features_fired == 0).nonzero()[:, 0]
        dead_features += features_fired == 0
        dead_features[features_fired != 0] = 0
        num_dead_features = (dead_features >= 5).sum()
        
        print(avg_features_fired)
        print(num_dead_features)
        
        net.dead_features = dead_features
        
    net.change_stage(2)
    PATH = opt.sae_model_path
    torch.save(net.state_dict(), PATH)
    
    return net

def getVis(visloader, net, opt):
    
    vis = torch.zeros((opt.batch_size, 16))
    net.eval()
    net.change_stage(1)
    # net.change_stage(0)
    
    for data in visloader:
        inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
        inputs.requires_grad = True
        
        acts, recons, fs, dead_acts = net(inputs)
        # output = net(inputs)
        
        top_fs = torch.topk(fs[0], 16)[0]
        
        # voxel_generator = PointToVoxel([0.4, 0.4, 0.4], [-5, -5, -5, 5, 5, 5], 
        #                                 3, 700, 5, inputs.device)
        
        # _, vx_coords, _, pt_vx_ids = voxel_generator.generate_voxel_with_id(inputs[0].T)
        # valid_mask = pt_vx_ids >= 0
        # pt_coords = vx_coords[pt_vx_ids[valid_mask]]
        
        for f in top_fs:
        # for f in output[0]:
            f.backward(retain_graph=True)

            pc = inputs[0].T.clone().detach()
            grads = torch.norm(inputs.grad[0], dim=0)
            
            saliency_map = torch.zeros((pc.size(0))).cuda()
            for ind in range(pc.size(0)):
                dists = pc[ind].unsqueeze(0).repeat(pc.size(0), 1) - pc
                dists = torch.norm(dists, dim=1)
                
                n_vx_inds = (dists <= 0.2).nonzero()
                n_vx_scores = torch.exp(-dists[n_vx_inds])
                n_vx_scores = n_vx_scores / n_vx_scores.sum()
                
                saliency_map[ind] = torch.sum(grads[n_vx_inds] * n_vx_scores)

            saliency_map = (saliency_map / saliency_map.max())
            saliency_map = saliency_map.clone().cpu().numpy()
        
            input = pc.cpu().detach().numpy()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(input)
            
            colors = np.zeros(input.shape)
            colors[:, 0] = saliency_map
            # colors[:, 1] = saliency_map
            colors[:, 2] = 1 - saliency_map
            
            cloud.colors = o3d.utility.Vector3dVector(colors)
        
    return vis

def getAEVis(visloader, encoder, decoder, sae, classifier, opt):
    
    encoder.eval()
    classifier.eval()
    sae.eval()
    # net.change_stage(0)
    
    count = 0
    
    for data in visloader:
        inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
        inputs.requires_grad = True
        
        latents = encoder(inputs)
        _latents, fs, dead_x = sae(latents)
        output = classifier(latents)
        
        top_fs, inds = torch.topk(fs[0], 16)
        
        for i in range(16):
            if top_fs[i] == 0:
                continue
            animateFeatureChange(sae, top_fs[i], decoder, latents, inds[i], i, object_name=str(data['mid'][0][4:]))
        
        count += 1
        # for f in top_fs:
        #     f.backward(retain_graph=True)

        #     pc = inputs[0].T.clone().detach()
        #     grads = torch.norm(inputs.grad[0], dim=0) ** (1/3)
            
        #     saliency_map = torch.zeros((pc.size(0))).cuda()
        #     for ind in range(pc.size(0)):
        #         dists = pc[ind].unsqueeze(0).repeat(pc.size(0), 1) - pc
        #         dists = torch.norm(dists, dim=1)
                
        #         n_vx_inds = (dists <= 0.2).nonzero()
        #         n_vx_scores = torch.exp(-dists[n_vx_inds])
        #         n_vx_scores = n_vx_scores / n_vx_scores.sum()
                
        #         saliency_map[ind] = torch.sum(grads[n_vx_inds] * n_vx_scores)

        #     saliency_map = (saliency_map / saliency_map.max())
        #     saliency_map = saliency_map.clone().cpu().numpy()
        
        #     input = pc.cpu().detach().numpy()
        #     cloud = o3d.geometry.PointCloud()
        #     cloud.points = o3d.utility.Vector3dVector(input)
            
        #     colors = np.zeros(input.shape)
        #     colors[:, 0] = saliency_map
        #     # colors[:, 1] = saliency_map
        #     colors[:, 2] = 1 - saliency_map
            
        #     cloud.colors = o3d.utility.Vector3dVector(colors)

def animateFeatureChange(sae, f, decoder, latents, index, rank, frames=60, object_name='test'):
    
    frames -= 1
    vector = sae.sae2.weight[:, index]
    prop = 0
    
    rank = str(rank).zfill(2)
    
    if not os.path.exists(f'images/{object_name}/{rank}_{index}_{f:.2f}'):
        os.makedirs(f'images/{object_name}/{rank}_{index}_{f:.2f}')
    
    while prop <= frames:
        test_latents = latents - vector * f * prop / frames
        with torch.no_grad():
            points = decoder(test_latents).cpu().detach().numpy()[0]
        
        if prop == 0:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)   
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            vis.add_geometry(cloud)
        cloud.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'images/{object_name}/{rank}_{index}_{f:.2f}/' + str(prop).zfill(3) + '.png')
        prop += 1
    
    vis.destroy_window()
    
    imgs = []
    for i in range(frames+1):
        imgs.append(cv2.imread(f'images/{object_name}/{rank}_{index}_{f:.2f}/' + str(i).zfill(3) + '.png'))
        
    height, width, _ = imgs[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'images/{object_name}/{rank}_{index}_{f:.2f}/video.avi', fourcc, frames // 4, (width, height))
    
    for img in imgs:
        video.write(img)
        
    cv2.destroyAllWindows()
    video.release()
    
    for i in range(frames+1):
        os.remove(f'images/{object_name}/{rank}_{index}_{f:.2f}/' + str(i).zfill(3) + '.png')

    
def evalNet(trainloader, testloader, net, opt):
    net.eval()
    net.change_stage(0)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            
            if opt.model == 'pointnet':
                outputs, _ = net(inputs)
            else:
                outputs = net(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 10000:
                break
        
        print(f'Accuracy of the network on the first 10000 train images: {100 * correct // total} %')
        train_acc = str(100 * correct // total) + '%'
        
        correct = 0
        total = 0
        for data in testloader:
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            
            if opt.model == 'pointnet':
                outputs, _ = net(inputs)
            else:
                outputs = net(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct +=(predicted == labels).sum().item()
            
        print(f'Accuracy of the network on the first 10000 test images: {100 * correct // total} %')
        test_acc = str(100 * correct // total) + '%'
        
    net.train()
    return train_acc, test_acc

def evalAENet(trainloader, testloader, encoder, classifier, opt):
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            
            latents = encoder(inputs)
            outputs = classifier(latents)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 10000:
                break
        
        print(f'Accuracy of the network on the first 10000 train images: {100 * correct // total} %')
        train_acc = str(100 * correct // total) + '%'
        
        correct = 0
        total = 0
        for data in testloader:
            inputs, labels = data['train_points'].cuda(0), data['cate_idx'].cuda(0)
            
            latents = encoder(inputs)
            outputs = classifier(latents)           
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct +=(predicted == labels).sum().item()
            
        print(f'Accuracy of the network on the first 10000 test images: {100 * correct // total} %')
        test_acc = str(100 * correct // total) + '%'
        
    classifier.train()
    return train_acc, test_acc