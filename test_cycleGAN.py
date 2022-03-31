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
from train_cycleGAN import ImageDataset


if __name__ ==  "__main__":    
    # Definition of variables
    batchSize = 1
    dataroot = 'datasets/photo2monet/'
    generator = 'cyclegan_models/netG_A2B.pth'
    size = 256 # size of the photo crop
    n_cpu = 8
    multi_gpu = False
    
    if multi_gpu:
        assert(torch.cuda.device_count()>1)
        num_of_gpus = torch.cuda.device_count()
        device_ids = range(num_of_gpus)
        cuda = torch.cuda.is_available()
    else:
        cuda = torch.cuda.is_available()
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # Networks
    netG_A2B = Generator(3, 3)
    netG_A2B.load_state_dict(torch.load(generator))
    netG_A2B.eval()
    
    if multi_gpu:
        netG_A2B = torch.nn.DataParallel(netG_A2B, device_ids=device_ids).cuda()
    else:
        if cuda:
            netG_A2B.cuda()

    # Dataset loader
    print("Loading data...")
    transforms_ = [ transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ]
    dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=batchSize, shuffle=False, num_workers=n_cpu)
    print("Finished lodaing dataset!")
    ###### Testing ######
    # Create output dirs if they don't exist
    if not os.path.exists('cyclegan_results/generated_monet'):
        os.makedirs('cyclegan_results/generated_monet')


    for i, batch in enumerate(dataloader):
        # Inputs & targets memory allocation
        bs = batch['A'].shape[0]
        input_A = Tensor(bs, 3, size, size)

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))

        # Generate output
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)

        # Save image files
        save_image(fake_B, 'cyclegan_results/generated_monet/%04d.jpg' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
        sys.stdout.write('\n')