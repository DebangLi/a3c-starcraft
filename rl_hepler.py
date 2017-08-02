from __future__ import division
import os
import time
import copy
import numpy as np 

from setproctitle import setproctitle

import torch
import torch.multiprocessing as mp 

from utils import set_random_seed

import multi_agent_starcraft_env as sc 

def build_env(args, port):
	#port = 11111
	env = sc.MultiAgentEnv(args.ip, port,speed=30)
	return env

def async_train(args, creat_agent, model, model_eval):
	setproctitle('{}:train[MASTER]'.format(args.name))
	counter = mp.Value('l', 0)

	def run_trainer(port, process_idx):
		setproctitle('{}:train[{}]'.format(args.name, process_idx))
		set_random_seed(np.random.randint(0, 2 ** 32))

		agent = creat_agent(port, process_idx)

		train_loop(counter, args, agent)

	processes = []
	for process_idx in range(args.n_processes):
		processes.append(mp.Process(target=run_trainer, args=(process_idx+11111, process_idx+1,)))

	for p in processes:
		p.start()
	for p in processes:
		p.join()


def train_loop(counter, args, agent):
	try:
		global_t = 0
		while True:
			agent.set_lr((args.n_steps - global_t - 1) / args.n_steps * args.lr)

			t = agent.act()

			#Get and increment the global counter
			with counter.get_lock():
				counter.value += t
				global_t = counter.value

			if global_t > args.n_steps:
				break

	except KeyboardInterrupt:
		agent.finish()
		raise

	agent.finish()