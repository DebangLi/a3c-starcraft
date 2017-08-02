from __future__ import division
import time 
import os
import numpy as np

import torch 
from  torch.autograd import Variable

from a3c import A3C

from a3c_model import A3CLSTM, A3CLSTM_commnet
from a3c_model import A3CMLP, PG
from optimizer import RMSpropAsync
import torch.optim as optim

from rl import EvalResult
from rl_helper import async_train, build_env

import utils
from utils import split_weight_bias

DISTANCE_FACTOR = 16

def build_master_model(n_actions, args):
	model = A3CLSTM(n_actions)
	#model = PG(n_actions)
	model.share_memory()

	weights, biases = split_weight_bias(model)
	opt = RMSpropAsync([{'params': weights}, {'params': biases, 'weight_deacy': 0}],
		lr = args.lr, eps = 1e-1, alpha=0.99, weight_decay=args.weight_decay)
	#opt = optim.Adam(model.parameters(), lr=0.01)
	opt.share_memory()

	return model, opt

def build_master_model_commnet(n_actions, args):
	model = A3CLSTM_commnet(n_actions)
	model.share_memory()
	weights, biases = split_weight_bias(model)

	opt = RMSpropAsync([{'params': weights}, {'params': biases, 'weight_deacy': 0}],
		lr = args.lr, eps = 1e-1, alpha=0.99, weight_decay=args.weight_decay)
	opt.share_memory()

	return model, opt

def a3c_train(args):

	# the number of networks' output
	n_actions = 13

	model, opt = build_master_model(n_actions, args)

	def creat_agent(port, process_idx):
		env = build_env(args, port, name='Train' + str(process_idx))
		return A3C(model, opt, env, args.t_max, 0.99, beta=args.beta, process_idx=process_idx)


	def model_eval_func(model, env, **params):
		return model_eval(args, model, env, **params)

	async_train(args, creat_agent, model, model_eval_func)

def  a3c_train_commnet(args):
	
	n_actions = 13

	model, opt = build_master_model_commnet(n_actions, args)

	def creat_agent(port, process_idx):
		env = build_env(args, port, name='Train' + str(process_idx))
		return A3C(model, opt, env, args.t_max, 0.99, beta=args.beta,process_idx=process_idx)

	def model_eval_func(model, env, **params):
		return model_eval_commnet(args, model, env, **params)

	async_train(args, creat_agent, model, model_eval_func)

def model_eval(args, model, env, random=True, vis=None):
	if vis is not None:
		vis, window_id, fps = vis
		frame_dur = 1.0 / fps
		last_time = time.time()

	rewards, start_time = 0, time.time()
	
	obs = env.reset()
	t = 0
	while True:
		nagent = len(obs) - np.sum(obs, axis=0, dtype=np.int32)[5]
		nenemy = len(obs) - nagent

		action = np.zeros([nagent, env.action_space.shape[0]])
		'''
		if nenemy == 0:
			break
			'''

		n = 0
		for i in range(len(obs)):
			if obs[i][5] == 0:
				input = np.zeros(68)
				enemy_table = -np.ones(5)
				input[0] = obs[i][1]
				input[1] = obs[i][2]
				input[2] = obs[i][3]
				input[3] = obs[i][4]
				input[4] = obs[i][5]
				k = 5
				ind_enemy = 0
				for j in range(len(obs)):
					if j != i:
						dis = utils.get_distance(obs[i][6], -obs[i][7], obs[j][6], -obs[j][7]) / DISTANCE_FACTOR - 1
						degree = utils.get_degree(obs[i][6], -obs[i][7], obs[j][6], -obs[j][7]) / 180
						input[k] = degree
						k += 1
						input[k] = dis
						k += 1
						for l in range(5):
							input[k] = obs[j][l+1]
							k += 1
						if obs[j][5] == 1:
							enemy_table[ind_enemy] = obs[j][0]
							ind_enemy += 1
				pout, _ = model.pi_and_v(Variable(torch.from_numpy(input).float().unsqueeze(0), volatile=True))
				#pout = model.pi_and_v(Variable(torch.from_numpy(input).float().unsqueeze(0), volatile=True))
				command_id = pout.action_indices[0] if random else pout.most_probable_actions[0]
				action[n][0] = obs[i][0]
				if command_id < 5:
					action[n][1] = 1
					action[n][4] = enemy_table[command_id]
				else:
					action[n][1] = -1
					if command_id < 10:
						action[n][2] = (float(command_id) - 5)/4
					else:
						action[n][2] = (float(command_id) - 13)/4
					action[n][3] = 1
				n += 1
		obs, reward, done, _ = env.step(action)
		#print(reward)
		rewards += reward
		if args.save_path is not None:
			with open(os.path.join(args.save_path, 'rewards_eval'), 'a+') as f:
				f.write('{}: {}\n'.format(t, rewards))
		#reward += reward
		if vis is not None and time.time() > last_time + frame_dur:
			pass
		if done:
			break
		t += 1
		if t > 501:
			break

	return EvalResult(rewards, time.time()-start_time)

def model_eval_commnet(args, model, env, random=True, vis=None):

	if vis is not None:
		vis, window_id, fps = vis
		frame_dur = 1.0 / fps
		last_time = time.time()

	rewards, start_time = 0, time.time()
	
	obs = env.reset()
	t = 0
	nagent_pre = len(obs) - np.sum(obs, axis=0, dtype=np.int32)[5]
	while True:
		nagent = len(obs) - np.sum(obs, axis=0, dtype=np.int32)[5]
		if nagent_pre != nagent:
			#print('reseting the model {} {}'.format(nagent, nagent_pre))
			model.reset_state()
		nagent_pre = nagent
		nenemy = len(obs) - nagent
		#print('nagent: {} | env,episodes_step: {}'.format(nagent, env.episode_steps))
		'''
		if nagent == 0 and env.episode_steps == 0:
			obs = env.reset()
			continue
		elif nagent == 0 and env.episode_steps != 0:
			break
		'''

		action = np.zeros([nagent, env.action_space.shape[0]])
		input = np.zeros([nagent, 68])
		#"enemy table"
		enemy_table = - np.ones(5)
		n = 0
		ind_enemy = 0
		for i in range(len(obs)):
			if obs[i][5] == 0:
				action[n][0] = obs[i][0]
				input[n][0] = obs[i][1]
				input[n][1] = obs[i][2]
				input[n][2] = obs[i][3]
				input[n][3] = obs[i][4]
				input[n][4] = obs[i][5]
				k = 5
				for j in range(len(obs)):
					if j != i:
						dis = utils.get_distance(obs[i][6], -obs[i][7], obs[j][6], -obs[j][7]) / DISTANCE_FACTOR - 1
						degree = utils.get_degree(obs[i][6], -obs[i][7], obs[j][6], -obs[j][7]) / 180
						input[n][k] = degree
						k += 1
						input[n][k] = dis
						k += 1
						for l in range(5):
							input[n][k] = obs[j][l+1]
							k += 1

				n += 1
			else:
				enemy_table[ind_enemy] = obs[i][0]
				ind_enemy += 1
		#print(input)
		if len(input) != 0:
			#print(input)
			pout, _ = model.pi_and_v(Variable(torch.from_numpy(input).float()))
		for i in range(nagent):
			command_id = pout.action_indices[i] if random else pout.most_probable_actions[i]
			if command_id < 5:
				action[i][1] = 1
				action[i][4] = enemy_table[command_id]
			else:
				action[i][1] = -1
				if command_id < 10:
					action[i][2] = (float(command_id) - 5)/4
				else:
					action[i][2] = (float(command_id) - 13)/4
				action[i][3] = 1
		obs, reward, done, _ = env.step(action)
		#print('Reward: {} | done: {}'.format(reward, done))
		rewards += reward
		if args.save_path is not None:
			with open(os.path.join(args.save_path, 'rewards_eval'), 'a+') as f:
				f.write('{}: {}\n'.format(t, rewards))
		#reward += reward
		if vis is not None and time.time() > last_time + frame_dur:
			pass
		if done:
			break
		t += 1
		if t > 501:
			break

	return EvalResult(rewards, time.time()-start_time)