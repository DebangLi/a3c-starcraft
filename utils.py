from __future__ import division
import random

import numpy as np

import torch

import math

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def split_weight_bias(model):
    weights, biases = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            biases.append(p)
        else:
            weights.append(p)
    return weights, biases

def ensure_shared_grads(model, shared_model, process_ids):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        #print('{}: share grads between shared model and model ....'.format(process_ids))

def get_degree(x1, y1, x2, y2):
    radians = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(radians)

def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def get_position(degree, distance, x1, y1):
    theta = math.pi / 2 - math.radians(degree)
    return x1 + distance * math.sin(theta), y1 + distance * math.cos(theta)

def print_progress(episodes, wins):
    print "Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (episodes, wins, wins/(episodes + 1e-6))