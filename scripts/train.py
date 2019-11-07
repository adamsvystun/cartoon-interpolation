"""
Modified from:
https://github.com/HyeongminLEE/pytorch-sepconv
"""

import argparse
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from src.sepconv import SepConvNet
from src.dataset import CartoonDataset

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main(options):
    db_dir = options.train

    if not os.path.exists(options.out_dir):
        os.makedirs(options.out_dir)
    result_dir = options.out_dir + '/result'
    ckpt_dir = options.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    logfile = open(options.out_dir + '/log.txt', 'w')
    logfile.write('batch_size: ' + str(options.batch_size) + '\n')

    total_epoch = options.epochs
    batch_size = options.batch_size

    dataset = CartoonDataset(db_dir)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # TestDB = Middlebury_other(options.test_input, options.gt)
    # test_output_dir = options.out_dir + '/result'

    if options.load_model is not None:
        checkpoint = torch.load(options.load_model)
        kernel_size = checkpoint['kernel_size']
        model = SepConvNet(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        kernel_size = options.kernel
        model = SepConvNet(kernel_size=kernel_size)

    logfile.write('kernel_size: ' + str(kernel_size) + '\n')

    if torch.cuda.is_available():
        model = model.cuda()

    max_step = train_loader.__len__()

    # model.eval()
    # TestDB.Test(model, test_output_dir, logfile, str(model.epoch.item()).zfill(3) + '.png')

    while True:
        if model.epoch.item() == total_epoch:
            break
        model.train()
        for batch_idx, (frame0, frame1, frame2) in enumerate(train_loader):
            frame0 = to_variable(frame0)
            frame1 = to_variable(frame1)
            frame2 = to_variable(frame2)
            loss = model.train_model(frame0, frame2, frame1)
            if batch_idx % 100 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format(
                    'Train Epoch: ', '[' + str(model.epoch.item()) + '/' + str(total_epoch) + ']',
                    'Step: ', '[' + str(batch_idx) + '/' + str(max_step) + ']',
                    'train loss: ', loss.item()
                ))
        model.increase_epoch()
        if model.epoch.item() % 1 == 0:
            torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size},
                       ckpt_dir + '/model_epoch' + str(model.epoch.item()).zfill(3) + '.pth')
            # model.eval()
            # TestDB.Test(model, test_output_dir, logfile, str(model.epoch.item()).zfill(3) + '.png')
            logfile.write('\n')

    logfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SepConv Pytorch')

    # parameters
    parser.add_argument('--train', type=str, default='./data/dataset/')
    parser.add_argument('--kernel', type=int, default=51)
    parser.add_argument('--out_dir', type=str, default='./logs/')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--load_model', type=str, default=None)
    # parser.add_argument('--test_input', type=str, default='./data/dataset/input')
    parser.add_argument('--gt', type=str, default='./data/dataset/gt')

    args = parser.parse_args()
    main(args)
