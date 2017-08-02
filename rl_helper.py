from __future__ import division
import os
import time
import copy
import numpy as np 


from setproctitle import setproctitle

from pycrayon import CrayonClient

import torch
import torch.multiprocessing as mp 

from utils import set_random_seed
from rl import EvalResult

import multi_agent_starcraft_env as sc 

def build_env(args, port, name=None):
	#port = 11111
	env = sc.MultiAgentEnv(args.ip, port, name=name, speed=0, frame_skip=2, max_episode_steps=500)
	return env

def async_train(args, creat_agent, model, model_eval):
	setproctitle('{}:train[MASTER]'.format(args.name))
	counter = mp.Value('l', 0)

	def run_trainer(port, process_idx):
		setproctitle('{}:train[{}]'.format(args.name, process_idx))
		set_random_seed(np.random.randint(0, 2 ** 32))

		agent = creat_agent(str(port), process_idx)

		train_loop(counter, args, agent)

	def run_evalator(port):
		setproctitle('{}:eval'.format(args.name))
		set_random_seed(np.random.randint(0, 2 ** 32))

		eval_loop(counter, args, model, model_eval, port)

	processes = []
	ports = ['11111', '11112', '11113', '11114', '11115', '11116', '11117', '11118', '11119']
	processes.append(mp.Process(target=run_evalator, args=(ports[0],)))
	for process_idx in range(args.n_processes):
		processes.append(mp.Process(target=run_trainer, args=(ports[process_idx+1], process_idx+1,)))

	for p in processes:
		p.start()
	for p in processes:
		p.join()


def train_loop(counter, args, agent):
	try:
		global_t = 0
		while True:
			#print(agent.process_idx)
			agent.set_lr((args.n_steps - global_t - 1) / args.n_steps * args.lr)

			t = agent.act(args)
			#t = agent.act_commnet(args)

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

def eval_loop(counter, args, shared_model, model_eval, port):
	try:
		SEC_PER_DAY = 24*60*60

		env = build_env(args, port, name='Eval  ')
		model = copy.deepcopy(shared_model)
		model.eval()

		'''
		cc = CrayonClient()
		names = cc.get_experiment_names()
		summaries = []
		for idx in range(args.n_eval):
			name = "{} [{}]".format(args.name, idx+1)
			#print(name)
			#print(names)
			if name in names:
				cc.remove_experiment(name)
			summaries.append(cc.create_experiment(name))
			'''

		max_reward = None
		save_condition = args.save_intervel

		rewards = []
		start_time = time.time()
		while True:
			# Sync with the shared model
			model.load_state_dict(shared_model.state_dict())
			# print(shared_model.parameters())
			# print('affine1')
			'''
			for param in shared_model.parameters():
				#print(param.norm())
				break
				if name in ['head']:
					for name2, module2 in module.named_children():
						print(name2)
						print(module2.parameters()[0,0])
			continue
			'''

			restart, eval_start_time, eval_start_step = False, time.time(), counter.value
			results = []
			for i in range(args.n_eval):
				#print('Model eval: {}'.format(i))
				model.reset_state()
				results.append(model_eval(model, env))

			eval_end_time, eval_end_step = time.time(), counter.value
			results = EvalResult(*zip(*results))
			rewards.append((counter.value, results.reward))

			local_max_reward = np.max(results.reward)
			if max_reward is None or max_reward < local_max_reward:
				max_reward = local_max_reward

			if local_max_reward >= max_reward:
				# Save model
				torch.save(model.state_dict(), os.path.join(args.model_path, 'best_model.pth'))

			time_since_start = eval_end_time - start_time
			day = time_since_start // SEC_PER_DAY
			time_since_start %= SEC_PER_DAY

			seconds_to_finish = (args.n_steps - eval_end_step) / (eval_end_step - eval_start_step + 1e-10)*(eval_end_time - eval_start_time)
			days_to_finish = seconds_to_finish // SEC_PER_DAY
			seconds_to_finish %= SEC_PER_DAY
			print("STEP:[{}|{}], Time: {}d {}, Finish in {}d {}".format(counter.value, args.n_steps,
				'%02d' % day, time.strftime("%Hh %Mm %Ss", time.gmtime(time_since_start)),
				'%02d' % days_to_finish, time.strftime("%Hh %Mm %Ss", time.gmtime(seconds_to_finish))))
			print('\tMax reward: {}, avg_reward: {}, std_reward: {}, min_reward: {}, max_reward: {}'.format(
				max_reward, np.mean(results.reward), np.std(results.reward), np.min(results.reward), local_max_reward))

			# Plot
			'''
			for summary, reward in zip(summaries, results.reward):
				#print(reward)
				summary.add_scalar_value('reward', reward, step=eval_start_step)
				'''
			
			if counter.value > save_condition or counter.value >= args.n_steps:
				save_condition += args.save_intervel
				torch.save(model.state_dict(), os.path.join(args.model_path, 'model_iter_{}.pth'.format(counter.value)))
				torch.save(model.state_dict(), os.path.join(args.model_path, 'model_latest.pth'))

				with open(os.path.join(args.save_path, 'rewards'), 'a+') as f:
					for record in rewards:
						f.write('{}: {}\n'.format(record[0], record[1]))
				del rewards[:]

			if counter.value >= args.n_steps:
				print('Evaluator Finished !!!')
				break
	except KeyboardInterrupt:
		torch.save(shared_model.state_dict(), os.path.join(args.model_path, 'model_latest.pth'))
		raise
