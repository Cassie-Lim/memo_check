import numpy as np
import torch
import torch.nn as nn
import torchvision

print('Running...')
# predict_dict = {}
predict_dict = []

# corruptions = [
#     'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
#     'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
#     'brightness', 'contrast', 'elastic_transform', 'pixelate',
#     'jpeg_compression'
# ]
corruptions = [
    'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression'
]
level = 1
for corruption in corruptions:
    name = "avg_" + corruption + "_" + str(level)
    tmp_predict = np.load(file="/data/yusun/lmy/memo/cifar-10-exps/predict/" +
                          name + ".npy")
    tmp_predict = np.average(tmp_predict, axis=1)
    predict_dict.append(tmp_predict)
    # predict_dict.append(np.load(file="/data/yusun/lmy/memo/cifar-10-exps/predict/" +
    #                       name + ".npy"))
predict_dict = np.array(predict_dict)
print(predict_dict.shape)
avg_predict_probs = np.average(predict_dict, axis=0)
teset = torchvision.datasets.CIFAR10(
    root="/data/yusun/lmy/CIFAR-10.1/datasets",
    train=False,
    download=True,
    transform=None)
correctness = []
for i in range(avg_predict_probs.shape[0]):
    _, label = teset[i]
    predict_label = np.argmax(avg_predict_probs[i])
    correct = 1 if predict_label == label else 0
    correctness.append(correct)
print(f'Acc for average: {(np.mean(correctness))*100:.2f}')
