import torch
import torch.nn.functional as F
from torchvision.models.vgg import vgg16


# vgg = vgg16(pretrained=True, progress=False)
# if torch.cuda.is_available():
#     vgg.cuda()


def nll_loss(output, target):
    return F.nll_loss(output, target)


# def perceptual_loss(predicted_image, true_image):
#     features_y = vgg(predicted_image)
#     features_x = vgg(true_image)
#     return F.mse_loss(features_y.relu2_2, features_x.relu2_2)


def interpolation_loss(predicted_frame, true_frame, beta, gamma):
    loss =  beta * F.l1_loss(predicted_frame, true_frame)
    # if gamma != 0:
    #     loss += gamma * perceptual_loss(predicted_frame, true_frame)
    return loss


def cycle_consistency_loss(predicted_frames, true_frames, alphas, beta, gamma):
    loss = torch.zeros(predicted_frames[0].size(0)).cuda()
    for alpha, predicted_frame, true_frame in zip(alphas, predicted_frames, true_frames):
        loss += alpha * interpolation_loss(predicted_frame, true_frame, beta, gamma)
    return loss
