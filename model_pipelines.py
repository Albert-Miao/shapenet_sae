import torch
def trainNet(trainloader, testloader, net, opt):
    for epoch in range(opt.num_model_epochs):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(0), data[1].cuda()
            
            net.optimizer.zero_grad()
            

            outputs = net(inputs)
            loss = net.criterion(outputs, labels)
            running_loss += loss.item()
            
            loss.backward()
            net.optimizer.step()
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / opt.super_batch_size:.3f}')
                
                running_loss = 0.0
                
    PATH = './03_12_25.pth'
    torch.save(net.state_dict(), PATH)
    
    net.change_stage(1)
    for epoch in range(opt.num_sae_epochs):
        running_recon_loss = 0.0
        running_l1_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(0), data[1].cuda()
            
            net.optimizer.zero_grad()
            
            acts, recons, fs = net(inputs)
            recon_loss = net.criterion(acts, recons)
            l1_loss = net.l1criterion(fs, torch.zeros_like(fs))
            
            running_recon_loss += recon_loss.item()
            running_l1_loss += l1_loss.item()
            
            loss = recon_loss + l1_loss
            
            loss.backward()
            net.optimizer.step()
            
            if i % opt.super_batch_size == opt.suepr_batch_size - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] recon loss: {running_recon_loss / opt.suepr_btach.size:.3f} ' + 
                      f'l1 loss: {running_l1_loss / opt.super_batch_size:.3f}')
                
                running_recon_loss = 0.0
                running_l1_loss = 0.0
                
    net.change_stage()
    PATH = './03_12_25_sae.pth'
    torch.save(net.state_dict(), PATH)
    
    return net
        
def evalNet(trainloader, testloader, net, opt):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data[0].cuda(), data[1].cuda()
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
            inputs, labels = data[0].cuda(), data[1].cuda()
            
            outputs = net(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item(0)
            
        print(f'Accuracy of the network on the first 10000 test images: {100 * correct // total} %')
        test_acc = str(100 * correct // total) + '%'
        
    net.train()
    return train_acc, test_acc