from __future__ import division
import copy
import os

import torch
import torch.autograd as autograd
from torch.autograd import Variable

from pycrayon import CrayonClient

import numpy as np 

import utils
from utils import ensure_shared_grads

DISTANCE_FACTOR = 16

class A3C(object):
	"""A3C for gym-starcraft"""
	def __init__(self, model, optimizer, env, t_max, gamma, beta=1e-2, process_idx=0,
		clip_reward=True, phi=lambda x: x, pi_loss_coef=1.0, v_loss_coef=0.25, keep_loss_scale_same=False):
		# Gloabally shared model
		self.shared_model = model
		# Thread specific model
		self.model = copy.deepcopy(self.shared_model)
		self.optimizer = optimizer
		self.env = env

		self.t_max = t_max
		self.gamma = gamma
		self.beta = beta
		self.process_idx = process_idx
		self.clip_reward = clip_reward
		self.phi = phi
		self.pi_loss_coef = pi_loss_coef
		self.v_loss_coef = v_loss_coef
		self.keep_loss_scale_same = keep_loss_scale_same

		#self.cc = CrayonClient()
		#self.sum = self.cc.create_experiment(str(self.process_idx))

		self.done = False

		self.reset_state()

	def reset_state(self):
		#print('Reseting the state when the eposide step is {}'.format(self.env.episode_steps))
		self.obs = self.env.reset()
		#print(len(self.obs))
		self.done = False

	def update_state(self):
		pass

	#def get_independent_input(self):
	def act(self, args):

		self.model.load_state_dict(self.shared_model.state_dict())
		self.model.train()
		if self.done:
			self.reset_state()

		log_probs, entropies, rewards, values, actions=[], [], [], [], []
		for _ in range(self.t_max):

			nagent = len(self.obs) - np.sum(self.obs, axis = 0, dtype = np.int32)[5]
			nenemy = len(self.obs) - nagent
			#print('Length of obs {}, number of agent is {}, number of enemy is {}').format(len(self.obs), nagent, nenemy)

			action = np.zeros([nagent, self.env.action_space.shape[0]])

			n = 0
			for i in range(len(self.obs)):
				if self.obs[i][5] == 0:
					# initilize the input for model(the independent) 5+7*9=68
					input = np.zeros(68)
					enemy_table = -np.ones(5)
					input[0] = self.obs[i][1]
					input[1] = self.obs[i][2]
					input[2] = self.obs[i][3]
					input[3] = self.obs[i][4]
					input[4] = self.obs[i][5]
					k = 5
					ind_enemy = 0
					for j in range(len(self.obs)):
						if j != i:
							dis = utils.get_distance(self.obs[i][6], -self.obs[i][7],
								self.obs[j][6], -self.obs[j][7]) / DISTANCE_FACTOR - 1
							degree = utils.get_degree(self.obs[i][6], -self.obs[i][7], 
								self.obs[j][6], -self.obs[j][7]) / 180
							input[k] = degree
							k += 1
							input[k] = dis
							k += 1
							input[k] = self.obs[j][1]
							k += 1
							input[k] = self.obs[j][2]
							k += 1
							input[k] = self.obs[j][3]
							k += 1
							input[k] = self.obs[j][4]
							k += 1
							input[k] = self.obs[j][5]
							k += 1
							if self.obs[j][5] == 1:
								enemy_table[ind_enemy] = self.obs[j][0]
								ind_enemy += 1

					pout, vout = self.model.pi_and_v(Variable(torch.from_numpy(input).float().unsqueeze(0)))

					action[n][0] = self.obs[i][0]
					command_id = pout.action_indices[0]
					#self.sum.add_scalar_value('command', command_id)
					with open(os.path.join(args.save_path, 'command_id'), 'a+') as f:
						f.write('{}\n'.format(command_id))
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
					log_probs.append(pout.sampled_actions_log_probs)
					entropies.append(pout.entropy)
					values.append(vout)
			self.obs, reward, done, _ = self.env.step(action)

			if action is not None:
				n = len(action)
			else:
				n = 0
			for i in range(n):
				rewards.append(reward)

			if done:
				self.done = done
				break
			if self.env.episode_steps == self.env.max_episode_steps:
				self.done = True
				break
		R = 0
		input_one_agent = self.obs2input()
		if not self.done and self.obs is not None and input_one_agent is not None:
			_, vout = self.model.pi_and_v(Variable(torch.from_numpy(input_one_agent).float().unsqueeze(0)))
			R = float(vout.data.numpy())
		else:
			self.model.reset_state()

		t = len(rewards)
		if t == 0:
			return t

		pi_loss, v_loss = 0, 0
		for i in reversed(range(t)):
			R = self.gamma * R + rewards[i]
			v = values[i]

			advantage = R - float(v.data.numpy()[0,0])
			#print('R:{}  v:{}'.format(R, v))
			# Accumulate gradients of policy
			log_prob = log_probs[i]
			entropy = entropies[i]
			# Log probability is increased proportionally to advantage
			pi_loss -= log_prob * advantage
			# Entropy is maximized
			pi_loss -= self.beta * entropy
			# Accumulate gradients of value function
			v_loss += (v - R).pow(2).div_(2)
			#self.sum.add_scalar_value('r', rewards[i])
			#self.sum.add_scalar_value('v', v.data[0,0])
			#self.sum.add_scalar_value('R', R)
			#self.sum.add_scalar_value('Advantage', advantage)
		if self.pi_loss_coef != 1.0:
			pi_loss *= self.pi_loss_coef

		if self.v_loss_coef != 1.0:
			v_loss *= self.v_loss_coef
		# Normalize the loss of sequences trunctated by terminal states
		if self.keep_loss_scale_same and t < self.t_max:
			factor = self.t_max / t
			pi_loss *= factor
			v_loss *= factor

		total_loss = pi_loss + v_loss
		#print('total_loss:{}'.format(total_loss))


		# Compute gradients using thread-specific model
		self.optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)
		# Copy the gradients to the globally shared model
		ensure_shared_grads(self.model, self.shared_model, self.process_idx)
		self.optimizer.step()

		self.model.unchain_backward()
		#self.sum.add_scalar_value('total_loss', total_loss.data[0,0])



		return t

	# commnet to take action
	def act_commnet(self, args):

		self.model.load_state_dict(self.shared_model.state_dict())
		self.model.train()
		if self.done:
			self.reset_state()
		log_probs, entropies, rewards, values, actions=[], [], [], [], []
		nagent_pre = len(self.obs) - np.sum(self.obs, axis = 0, dtype = np.int32)[5]
		for _ in range(self.t_max):
			nagent = len(self.obs) - np.sum(self.obs, axis = 0, dtype = np.int32)[5]
			if nagent != nagent_pre:
				self.model.reset_state()
			nagent_pre = nagent
			nenemy = len(self.obs) - nagent
			'''
			if nagent == 0:
				self.done = True
				break
				'''

			action = np.zeros([nagent, self.env.action_space.shape[0]])
			input = np.zeros([nagent, 68])
			#"enemy table"
			enemy_table = - np.ones(5)
			n = 0
			ind_enemy = 0
			for i in range(len(self.obs)):
				if self.obs[i][5] == 0:
					action[n][0] = self.obs[i][0]
					input[n][0] = self.obs[i][1]
					input[n][1] = self.obs[i][2]
					input[n][2] = self.obs[i][3]
					input[n][3] = self.obs[i][4]
					input[n][4] = self.obs[i][5]
					k = 5
					for j in range(len(self.obs)):
						if j != i:
							dis = utils.get_distance(self.obs[i][6], -self.obs[i][7],
								self.obs[j][6], -self.obs[j][7]) / DISTANCE_FACTOR - 1
							degree = utils.get_degree(self.obs[i][6], -self.obs[i][7], 
								self.obs[j][6], -self.obs[j][7]) / 180
							input[n][k] = degree
							k += 1
							input[n][k] = dis
							k += 1
							input[n][k] = self.obs[j][1]
							k += 1
							input[n][k] = self.obs[j][2]
							k += 1
							input[n][k] = self.obs[j][3]
							k += 1
							input[n][k] = self.obs[j][4]
							k += 1
							input[n][k] = self.obs[j][5]
							k += 1
					n += 1
				else:
					enemy_table[ind_enemy] = self.obs[i][0]
					ind_enemy += 1
			if len(input) != 0:
				pout, vout = self.model.pi_and_v(Variable(torch.from_numpy(input).float()))
				assert vout.size(0) == nagent
				assert pout.logits.size(0) == nagent

			for i in range(nagent):
				command_id = pout.action_indices[i]
				if args.save_path is not None:
					with open(os.path.join(args.save_path, 'command_id'), 'a+') as f:
						f.write('{}\n'.format(command_id))
				#self.sum.add_scalar_value('command', command_id)
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
				
				log_probs.append(pout.sampled_actions_log_probs[i])
				entropies.append(pout.entropy[i])
				values.append(vout[i])

			self.obs, reward, done, _ = self.env.step(action)

			if action is not None:
				n = len(action)
			else:
				n = 0
			for i in range(n):
				rewards.append(reward)

			if done:
				self.done = done
				break
			if self.env.episode_steps == self.env.max_episode_steps:
				self.done = True
				break
		R = 0

		input_all_agent = self.obs2input2()
		if not self.done and len(self.obs) != 0 and len(input_all_agent) != 0:
			if len(input_all_agent) != nagent_pre:
				self.model.reset_state()
			_, vout = self.model.pi_and_v(Variable(torch.from_numpy(input_all_agent).float()))
			R = float(vout.data.numpy()[0])
		else:
			self.model.reset_state()

		t = len(rewards)
		if t == 0:
			return t

		pi_loss, v_loss = 0, 0
		for i in reversed(range(t)):
			R = self.gamma * R + rewards[i]
			v = values[i]

			advantage = R - float(v.data.numpy()[0])
			#print('R:{}  v:{}'.format(R, v))
			# Accumulate gradients of policy
			log_prob = log_probs[i]
			entropy = entropies[i]
			# Log probability is increased proportionally to advantage
			pi_loss -= log_prob * advantage
			# Entropy is maximized
			pi_loss -= self.beta * entropy
			# Accumulate gradients of value function
			v_loss += (v - R).pow(2).div_(2)

		if self.pi_loss_coef != 1.0:
			pi_loss *= self.pi_loss_coef

		if self.v_loss_coef != 1.0:
			v_loss *= self.v_loss_coef
		# Normalize the loss of sequences trunctated by terminal states
		if self.keep_loss_scale_same and t < self.t_max:
			factor = self.t_max / t
			pi_loss *= factor
			v_loss *= factor

		total_loss = pi_loss + v_loss
		#print('total_loss:{}'.format(total_loss))


		# Compute gradients using thread-specific model
		self.optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)
		# Copy the gradients to the globally shared model
		ensure_shared_grads(self.model, self.shared_model, self.process_idx)
		self.optimizer.step()

		self.model.unchain_backward()
		#self.sum.add_scalar_value('total_loss', total_loss.data[0])



		return t






	def set_lr(self, lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def finish(self):
		print('Trainer [{}] Finished !!!'.format(self.process_idx))


	def obs2input(self):
		n = 0
		for i in range(len(self.obs)):
			if self.obs[i][5] == 0:
				#initilize the input to the model 5+7*9 = 68
				input = np.zeros(68)
				enemy_table = -np.ones(5)
				input[0] = self.obs[i][1]
				input[1] = self.obs[i][2]
				input[2] = self.obs[i][3]
				input[3] = self.obs[i][4]
				input[4] = self.obs[i][5]
				k = 5
				ind_enemy = 0
				for j in range(len(self.obs)):
					if j != i:
						dis = utils.get_distance(self.obs[i][6], -self.obs[i][7], self.obs[j][6], -self.obs[j][7]) / DISTANCE_FACTOR - 1
						degree = utils.get_degree(self.obs[i][6], -self.obs[i][7], self.obs[j][6], -self.obs[j][7]) / 180
						input[k] = degree
						k += 1
						input[k] = dis
						k += 1
						input[k] = self.obs[j][1]
						k += 1
						input[k] = self.obs[j][2]
						k += 1
						input[k] = self.obs[j][3]
						k += 1
						input[k] = self.obs[j][4]
						k += 1
						input[k] = self.obs[j][5]
						k += 1

				return input

	def obs2input2(self):
		nagent = len(self.obs) - np.sum(self.obs, axis = 0, dtype = np.int32)[5]
		nenemy = len(self.obs) - nagent
		input = np.zeros([nagent, 68])
		n = 0
		for i in range(len(self.obs)):
			if self.obs[i][5] == 0:
				input[n][0] = self.obs[i][1]
				input[n][1] = self.obs[i][2]
				input[n][2] = self.obs[i][3]
				input[n][3] = self.obs[i][4]
				input[n][4] = self.obs[i][5]
				k = 5
				for j in range(len(self.obs)):
					if j != i:
						dis = utils.get_distance(self.obs[i][6], -self.obs[i][7],
							self.obs[j][6], -self.obs[j][7]) / DISTANCE_FACTOR - 1
						degree = utils.get_degree(self.obs[i][6], -self.obs[i][7], 
							self.obs[j][6], -self.obs[j][7]) / 180
						input[n][k] = degree
						k += 1
						input[n][k] = dis
						k += 1
						input[n][k] = self.obs[j][1]
						k += 1
						input[n][k] = self.obs[j][2]
						k += 1
						input[n][k] = self.obs[j][3]
						k += 1
						input[n][k] = self.obs[j][4]
						k += 1
						input[n][k] = self.obs[j][5]
						k += 1
				n += 1

		return input






