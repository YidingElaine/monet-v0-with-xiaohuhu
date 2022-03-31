#  CycleGAN (https://arxiv.org/abs/1703.10593) implementation
#  https://github.com/aitorzip/PyTorch-CycleGAN

import sys
import os

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from train_cycleGAN import Generator
from saliency_v_bp import *

if __name__ ==  "__main__":   
    # transform to monet 
    # Definition of variables
    batch_size = 1
    size = 224
    generator = 'cyclegan_models/netG_A2B.pth'
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # Networks
    netG_A2B = Generator(3, 3)
    netG_A2B.load_state_dict(torch.load(generator))
    netG_A2B.eval()

    if cuda:
        netG_A2B.cuda()

    # Dataset loader
    print("Loading data...")
    data_transform = transforms.Compose([
        transforms.Resize((size, size)),
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
    
    target_transform = transforms.Lambda(lambda y: change_label(y))
    test_dataset = datasets.ImageFolder(root='images', transform=data_transform, target_transform=target_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Finished lodaing dataset!")
    
    # Create output dirs if they don't exist
    for i in range(5):
        if not os.path.exists(f"cyclegan_results/generated_monet_imageNet/{i}"):
            os.makedirs(f"cyclegan_results/generated_monet_imageNet/{i}")

    for i, (data, _) in enumerate(test_loader):
        # Inputs & targets memory allocation
        bs = data.shape[0]
        input_A = Tensor(bs, 3, size, size)

        # Set model input
        real_A = Variable(input_A.copy_(data))

        # Generate output
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)

        # Save image files
        save_image(fake_B, 'cyclegan_results/generated_monet_imageNet/%d/%04d.jpg' %((i//3), (i+1)))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(test_loader)))
        sys.stdout.write('\n')
    
    # generate saliency maps    
    # paths
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    if not os.path.exists('saliency_maps/monet/smap+image'):
        os.makedirs('saliency_maps/monet/smap+image')
    if not os.path.exists('saliency_maps/monet/smap'):
        os.makedirs('saliency_maps/monet/smap')
    save_path = PATH + 'saliency_maps/monet/'

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    data_dataset = datasets.ImageFolder(root='cyclegan_results/generated_monet_imageNet', transform=data_transform, target_transform=target_transform)
    data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    saliency_method = InputGradient(model)
    unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    for batch_idx, (data, label) in enumerate(data_loader):
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