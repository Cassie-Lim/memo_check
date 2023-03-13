import pickle
import numpy as np
import sys
import os
import cv2
from utils.do_corruptions import Corruption
import torchvision.transforms as transforms

# corruptions = [
#     'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
#     'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
#     'brightness', 'contrast', 'elastic_transform', 'pixelate',
#     'jpeg_compression'
# ]
corruptions = ['gaussian_noise', 'impulse_noise', 'glass_blur', 'motion_blur']

orig_data = []
downloaded_list = [
    # 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
    # 'data_batch_5', 'test_batch'
    'test_batch'
]

# now load the picked numpy arrays
for file_name in downloaded_list:
    file_path = os.path.join(
        '/data/yusun/lmy/CIFAR-10.1/datasets/cifar-10-batches-py', file_name)
    with open(file_path, 'rb') as f:
        if sys.version_info[0] == 2:
            entry = pickle.load(f)
        else:
            entry = pickle.load(f, encoding='latin1')
        orig_data.append(entry['data'])
orig_data = np.vstack(orig_data).reshape(-1, 3, 32, 32)
orig_data = orig_data.transpose((0, 2, 3, 1))  # convert to HWC

cv2.imwrite("./images_sample/original.png", orig_data[0])
for op in corruptions:
    print("Applying {} to images...".format(op))
    corruptor = Corruption(op)
    new_data = [[], [], [], [], []]
    for img in orig_data:
        for level in range(5):
            corrupted_img = corruptor(img, level + 1)
            # # normalize
            # corrupted_img = np.divide(
            #     corrupted_img - np.array([0.5, 0.5, 0.5]),
            #     np.array([0.5, 0.5, 0.5]))
            new_data[level].append(corrupted_img)

    for level in range(5):
        cv2.imwrite(f"./images_sample/{op}_{level}.png", new_data[level][0])
        np.save(
            '/data/yusun/lmy/CIFAR-10c/' + op + '_' + str(level + 1) + '.npy',
            new_data[level])
    print("Images processed with {} have been saved".format(op))
