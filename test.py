import argparse
import time
import multi_agent_starcraft_env as sc
import utils
from itertools import count
from collections import namedtuple 
from a3c_model import A3CLSTM, A3CLSTM_commnet
from a3c_helper import model_eval, model_eval_commnet

import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Independent agent to fight')
parser.add_argument('--seed', type=int, default=543, metavar='N',
					help='random seed (default: 543)')
parser.add_argument('--ip', help='server ip')
parser.add_argument('--save_path', help='save path', default=None)
parser.add_argument('--port', help='server port', default="11111")
parser.add_argument('--model', default='', help="path to model (to continue training)")
args = parser.parse_args()
print(args)

DISTANCE_FACTOR = 16


if __name__ == '__main__':

	print('Inilizing the model...')
	env = sc.MultiAgentEnv(args.ip, args.port, speed=0, max_episode_steps=500)
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	model = A3CLSTM(13)
	#model = A3CLSTM_commnet(13)
	if args.model !='':
		model.load_state_dict(torch.load(args.model))
		print('Loading model:')
		print(model)

	for i in range(201):

		print('The {}th time to test.'.format(i))
		model_eval(args, model, env)
		
		#model_eval_commnet(args, model, env)
		#model.reset_state()




