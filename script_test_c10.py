import sys

sys.path.insert(0, '../')
import argparse

from subprocess import call

parser = argparse.ArgumentParser()
# parser.add_argument('--experiment', default='cifar10c')
parser.add_argument('--experiment', default='cifar101')
parser.add_argument('--resume', default='rn26_gn')
args = parser.parse_args()
experiment = args.experiment
resume = args.resume

dataroot = '/data/yusun/lmy/'  # EDIT THIS
# dataroot = '/data/yusun/lmy/CIFAR-10.1/datasets'  # EDIT THIS
# dataroot = '/data/yusun/lmy/CIFAR-10c'

if experiment == 'cifar10':
    corruptions = ['original']
    levels = [0]
elif experiment == 'cifar101':
    corruptions = ['cifar_new']
    levels = [0]
elif experiment == 'cifar10c':
    # corruptions = [
    #     'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    #     'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    #     'brightness', 'contrast', 'elastic_transform', 'pixelate',
    #     'jpeg_compression'
    # ]
    corruptions = [
        'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    # levels = [1]
    levels = [1, 2, 3, 4, 5]

    # corruptions = [
    #     'gaussian_noise', 'impulse_noise', 'glass_blur', 'motion_blur'
    # ]
    # corruptions = ['gaussian_noise']
    # levels = [1]

# for corruption in corruptions:
#     for level in levels:
#         print(corruption, level)
#         call(' '.join([
#             'python', 'test_calls/test_initial.py', f'--dataroot {dataroot}',
#             f'--level {level}', f'--corruption {corruption}',
#             f'--resume results/cifar10_{resume}/'
#         ]),
#              shell=True)

#         call(' '.join([
#             'python', 'test_calls/test_adapt.py', f'--dataroot {dataroot}',
#             f'--level {level}', f'--corruption {corruption}',
#             f'--resume results/cifar10_{resume}/'
#         ]),
#              shell=True)

for corruption in corruptions:
    for level in levels:
        print(corruption, level)
        call(' '.join([
            'python', 'test_calls/test_my.py', f'--dataroot {dataroot}',
            f'--level {level}', f'--corruption {corruption}',
            f'--resume results/cifar10_{resume}/'
        ]),
             shell=True)