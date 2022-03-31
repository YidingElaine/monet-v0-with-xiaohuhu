# Saliency via Backprop (https://arxiv.org/abs/1312.6034) implementation
# https://github.com/idiap/fullgrad-saliency/blob/master/saliency/grad.py
import os
import cv2
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms

# Algorithm

class InputGradient():
    """
    Compute input-gradient saliency map 
    """

    def __init__(self, model, loss=False):
        # loss -> compute gradients w.r.t. loss instead of 
        # gradients w.r.t. logits (default: False)
        self.model = model
        self.loss = loss

    def _getGradients(self, image, target_class=None):
        """
        Compute input gradients for an image
        """

        image = image.requires_grad_()
        #outputs = torch.log_softmax(self.model(image),1)
        outputs = self.model(image)

        if target_class is None:
            target_class = (outputs.data.max(1, keepdim=True)[1]).flatten()

        if self.loss:
            outputs = torch.log_softmax(outputs, 1)
            agg = F.nll_loss(outputs, target_class, reduction='sum')
        else:
            agg = -1. * F.nll_loss(outputs, target_class, reduction='sum')

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        gradients = torch.autograd.grad(outputs = agg, inputs = image, only_inputs=True, retain_graph=False)[0]

        # First element in the feature list is the image
        return gradients

    def saliency(self, image, target_class=None):

        self.model.eval()
        input_grad = self._getGradients(image, target_class=target_class)
        return torch.abs(input_grad).sum(1, keepdim=True)

# Utils

class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())


def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None

def save_saliency_map(image, saliency_map, filename1, filename2):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224,224))

    image = np.uint8(image * 255).transpose(1, 2, 0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    
    cv2.imwrite(filename1, np.uint8(255 * img_with_heatmap))
    cv2.imwrite(filename2, np.uint8(saliency_map))

if __name__ == "__main__":
    
    # paths
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    if not os.path.exists('saliency_maps/smap+image'):
        os.makedirs('saliency_maps/smap+image')
    if not os.path.exists('saliency_maps/smap'):
        os.makedirs('saliency_maps/smap')
    save_path = PATH + 'saliency_maps/'
    # cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if cuda else "cpu")
    batch_size = 3
    # Dataset loader for sample images
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def change_label(input):
        if isinstance(input, int):
            if input == 0:
                output = 0
            elif input == 1:
                output = 15
            elif input == 2:
                output = 173
            elif input == 3:
                output = 247
            elif input == 4:
                output = 665
        return output
    # class_to_idx = {'n01440764': 0, 'n01558993': 15, 'n02091244': 173, 'n02109525': 247, 'n03785016': 665}
    target_transform = transforms.Lambda(lambda y: change_label(y))
    test_dataset = datasets.ImageFolder(root='images', transform=data_transform, target_transform=target_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    
    saliency_method = InputGradient(model)
    unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    for batch_idx, (data, label) in enumerate(test_loader):
        # data = data.to(device).requires_grad_()
        data = data.requires_grad_()
        # Compute saliency maps for the input data
        saliency_map = saliency_method.saliency(data, target_class=label)
        # Save saliency maps
        for i in range(data.size(0)):
            filename1 = save_path + 'smap+image/' + str(batch_idx * batch_size + i + 1)
            filename2 = save_path + 'smap/' + str(batch_idx * batch_size + i + 1)
            image = unnormalize(data[i].cpu())

            save_saliency_map(image, saliency_map[i], filename1 + '.jpg', filename2 + '.jpg')
            
    print('Saliency maps saved.')