import torch
import numpy as np


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def psnr(output, target):
    with torch.no_grad():
        original = target.cpu().detach().numpy()
        compared = output.cpu().detach().numpy()
        mse = np.mean(np.square(original - compared))
        value = np.clip(
            np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    return value


def ssim():
    pass