import argparse
import sys
import time
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torchvision
from tqdm import tqdm
from models import *
import matplotlib.pyplot as plt
import os

random_seed = 1
torch.manual_seed(random_seed) #设置随机种子，可以使每次生成的随机参数都一样
torch.cuda.manual_seed(random_seed)
#torch.cuda.manual_seed_all() #为所有gpu设置随机种子
train_losses = []
train_counter = []
test_losses = []
# test_counter = [i*len(train_loader.dataset) for i in range(100 + 1)]

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, default='train')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument(
        '--learning_rate_decay_frequency', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--momentum',type=float,default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_iters', type=int, default=100000)
    parser.add_argument('--loss', type=str, default='soft_triplet')
    parser.add_argument('--loader_num_workers', type=int, default=4)
    args = parser.parse_args()
    return args

def train(opt,model,optimizer,trainset,logger,testset=None):
    iter=0
    epoch=-1 
    tic = time.time()  
    while iter<opt.num_iters:
        epoch+=1
        print('It', iter, 'epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                          4), opt.comment)
        train_loader=torch.utils.data.DataLoader(
            trainset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.loader_num_workers)
        model.train()
        for data, target in tqdm(train_loader, desc='Training for epoch ' + str(epoch)):
            iter += 1
            data=data.cuda()
            target=target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            logger.add_scalar('loss',loss.item(),global_step=iter)
            if iter >= opt.learning_rate_decay_frequency and iter % opt.learning_rate_decay_frequency == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
        if epoch%3==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, iter, opt.num_iters,
            100. * iter / opt.num_iters, loss.item()))
            train_losses.append(loss.item())
            
            # train_counter.append((batch_idx*len(data)) + ((opt.num_iters-1)*len(train_loader.dataset)))
            torch.save({'iter':iter,'opt':opt,'model_state_dict':model.state_dict()}, logger.file_writer.get_logdir() +'/model.pth')
            torch.save(optimizer.state_dict(), logger.file_writer.get_logdir() +'/optimizer.pth')
        
            test(opt,model,testset,logger)

def test(opt,model,testset,logger):
    test_loader=torch.utils.data.DataLoader(
        testset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.loader_num_workers)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.cuda()
            target=target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def showImage(startIndex,lastIndex,dataset):
    data_loader=torch.utils.data.DataLoader(
        dataset,
        batch_size=6,
        shuffle=True,
        drop_last=True,
        num_workers=5)
    print(data_loader.dataset)
    examples=enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def loadmodel(model,optimizer,path):
    model_state_dict = torch.load(os.path.join(path,'model.pth'))
    model.load_state_dict(model_state_dict['model_state_dict'])
    optimizer_state_dict = torch.load(os.path.join(path,'optimizer.pth'))
    optimizer.load_state_dict(optimizer_state_dict)
    return model,optimizer

def main():
    opt=parse_opt()
    logger = SummaryWriter(comment=opt.comment)
    trainset=torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

    testset=torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

    model = Net()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate,
                      momentum=opt.momentum)
    # model,optimizer=loadmodel(model,optimizer,"./runs/Dec03_20-13-01_zzm-ThinkStation-P318train/")
    test(opt,model,testset,logger)
    train(opt,model,optimizer,trainset,logger,testset=testset)
    # showImage(1,6,trainset)
    logger.close()


if __name__ == '__main__':
    main()
