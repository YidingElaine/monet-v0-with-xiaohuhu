# quantitative comparisons 

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        else:  
            list_name.append(file_path) 
            
smap_path1 = 'saliency_maps/smap'
smap_path2 = 'saliency_maps/monet/smap'
smap_dir1 = []
smap_dir2 = []
if not os.path.exists('saliency_maps/difference'):
    os.makedirs('saliency_maps/difference')
save_path = 'saliency_maps/difference/'
listdir(smap_path1, smap_dir1)
listdir(smap_path2, smap_dir2)
smap_dir1.sort()
smap_dir2.sort()
ssim_score_list = []
mse_list = []
mean_original = []
mean_monet = []

for i in range(15):
    # load the two input images
    im1 = cv2.imread(smap_dir1[i])
    im2 = cv2.imread(smap_dir2[i])
    # # convert the images to grayscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # qualitative comparisons 
    im3 = np.zeros((224,224),dtype='uint8')
    im3 = cv2.addWeighted(im2, -1, im3, 0, 255, 0)
    oppsite = cv2.addWeighted(im1, 0.5, im3, 0.5, 0).astype("uint8")
    cv2.imwrite(save_path + f"opp_{i}.jpg", oppsite)
    # quantitative
    # compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned

    (score, _) = compare_ssim(im1, im2,gradient=True,channel_axis=2)
    ssim_score_list.append(score)
    mse = np.mean((im1 - im2) ** 2)
    mse_list.append(mse)
    print(f"[image{i}]SSIM: {score} | MSE: {mse}")
    mean_original.append(np.mean(im1 ** 2))
    mean_monet.append(np.mean(im2 ** 2))
    
plt.figure(1)
plt.plot(range(15), ssim_score_list,'-x')
plt.xlabel("Image")
plt.ylabel("SSIM score of different saliency")
plt.xticks(range(0,15,1))
plt.title("SSIM score of different images")
plt.savefig(save_path + 'ssim.jpg')

plt.figure(2)
plt.plot(range(15), mse_list,'-x',label='mean square error')
plt.plot(range(15), mean_original,'-',label='orignial mean square')
plt.plot(range(15), mean_monet,'-',label='orignial mean square')
plt.xlabel("Image")
plt.ylabel("MSE of different saliency")
plt.legend()
plt.xticks(range(0,15,1))
plt.title("Mean Square Error of different images")
plt.savefig(save_path + 'mse.jpg')