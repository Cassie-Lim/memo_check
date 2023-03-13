from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from PIL import Image
from tqdm import tqdm

from utils.prepare_dataset import prepare_test_data, te_transforms
from utils.test_helpers import build_model
from utils.third_party import aug

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', required=True)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--group_norm', default=8, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--resume', required=True)
args = parser.parse_args()

net = build_model(args)
teset, teloader = prepare_test_data(args)

print(f'Resuming from {args.resume}...')
ckpt = torch.load(args.resume + '/ckpt.pth')


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


optimizer = optim.SGD(net.parameters(), lr=args.lr)


def adapt_single(image):
    net.eval()
    for iteration in range(args.niter):
        inputs = [aug(image) for _ in range(args.batch_size)]
        inputs = torch.stack(inputs).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss, logits = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()


def test_single(model, image, label):
    model.eval()
    inputs = te_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs.cuda())
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs,
                                           dim=1).squeeze()[predicted].item()
    correctness_avg = 1 if predicted.item() == label else 0
    return correctness_avg, confidence


def predict_label_single(model, image, label):
    # return label & acc
    model.eval()
    inputs = te_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs.cuda())
        _, predicted = outputs.max(1)
    correctness_avg = 1 if predicted.item() == label else 0
    return correctness_avg, outputs


def predict_label_avg(model, image, label):
    # return label & acc
    model.eval()
    inputs = [aug(image) for _ in range(args.batch_size)]
    inputs = torch.stack(inputs).cuda()
    with torch.no_grad():
        outputs = model(inputs.cuda())
        _, predicted = outputs.mean(dim=0, keepdim=True).max(1)
    correctness_avg = 1 if predicted.item() == label else 0
    return correctness_avg, outputs


# ----------------------------------------baseline test----------------------------------------------
correctness_base = []
predicted_probs = []
net.load_state_dict(ckpt['net'])
for i in tqdm(range(len(teset))):
    _, label = teset[i]
    image = Image.fromarray(teset.data[i])
    correct, outputs = predict_label_single(net, image, label)
    correctness_base.append(correct)
    predicted_probs.append(outputs.cpu().detach().numpy())
test_err_base = (1 - np.mean(correctness_base)) * 100
print(
    f'Baseline test error for {args.corruption} {args.level}: {test_err_base:.2f}'
)

# ----------------------------------------avg prediction test----------------------------------------------
correctness_avg = []
predicted_probs = []
net.load_state_dict(ckpt['net'])
for i in tqdm(range(len(teset))):
    _, label = teset[i]
    image = Image.fromarray(teset.data[i])
    correct, outputs = predict_label_avg(net, image, label)
    correctness_avg.append(correct)
    # predicted_labels.append(outputs)
"""
    predicted_probs.append(outputs.cpu().detach().numpy())
np.save(file="/data/yusun/lmy/memo/cifar-10-exps/predict/" + "avg_" +
        args.corruption + "_" + str(args.level) + ".npy",
        arr=np.array(predicted_probs))
"""
# print(
#     f'Acc for {args.corruption} {args.level}:{(np.mean(correctness_avg))*100:.2f}')
test_err_avg = (1 - np.mean(correctness_avg)) * 100
print(
    f'Averaging predictions test error for {args.corruption} {args.level}: {test_err_avg:.2f}'
)

# ----------------------------------------memo test----------------------------------------------
correctness_memo = []
for i in tqdm(range(len(teset))):
    net.load_state_dict(ckpt['net'])
    _, label = teset[i]
    image = Image.fromarray(teset.data[i])

    adapt_single(image)
    correctness_memo.append(test_single(net, image, label)[0])
test_err_memo = (1 - np.mean(correctness_memo)) * 100
print(
    f'MEMO adapt test error for {args.corruption} {args.level}: {test_err_memo:.2f}'
)

test_err = np.array([test_err_base, test_err_avg, test_err_memo])
np.save(file="/data/yusun/lmy/memo/cifar-10-exps/test_err/" + "err_" +
        args.corruption + "_" + str(args.level) + ".npy",
        arr=np.array(test_err))
