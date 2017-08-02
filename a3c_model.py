from __future__ import division
import torch
import torch.nn as nn

from torch.autograd import Variable

from modules import IndependentHead, FCSoftmaxPolicy, FCVFunction, CommnetHead
class A3CModel(object):
	def pi_and_v(self, state, keep_same_state=False):
		raise NotImplementedError()

		def reset_state(self):
			pass

		def unchain_backward(self):
			pass

class A3CLSTM(nn.Module, A3CModel):
	def __init__(self, n_actions):
		super(A3CLSTM, self).__init__()

		self.head = IndependentHead()
		self.pi = FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
		self.v = FCVFunction(self.head.n_output_channels)
		self.lstm = nn.LSTMCell(self.head.n_output_channels, self.head.n_output_channels)
		self.reset_state()

	def reset_state(self):
		self.un_init = True

	def pi_and_v(self, state, keep_same_state=False):
		if self.un_init:
			batch_size = state.size()[0]
			self.h, self.c = Variable(torch.zeros(batch_size, self.head.n_output_channels)), Variable(torch.zeros(batch_size, self.head.n_output_channels))
			self.un_init = False

		out = self.head(state)
		h, c = self.lstm(out, (self.h, self.c))
		if not keep_same_state:
			self.h, self.c = h, c
		return self.pi.compute_policy(h), self.v(h)

	def unchain_backward(self):
		if self.un_init:
			return
		self.h.detach_()
		self.c.detach_()

class A3CLSTM_commnet(nn.Module, A3CModel):
	def __init__(self, n_actions):
		super(A3CLSTM_commnet, self).__init__()

		# n * 128
		self.head = CommnetHead()
		self.pi = FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
		self.v = FCVFunction(self.head.n_output_channels)
		self.lstm = nn.LSTMCell(self.head.n_output_channels, self.head.n_output_channels)
		self.reset_state()

	def reset_state(self):
		self.un_init = True

	def pi_and_v(self, state, keep_same_state=False):
		if self.un_init:
			# number of agent
			#print('State.size: {}'.format(state.size()))
			batch_size = state.size()[0]
			self.h, self.c = Variable(torch.zeros(batch_size, self.head.n_output_channels)), Variable(torch.zeros(batch_size, self.head.n_output_channels))
			self.un_init = False
		out = self.head(state)
		#print('-----------------------')
		#print(out.size())
		#print(self.h.size())
		#print(self.c.size())
		#print('------------------------')
		h, c = self.lstm(out, (self.h, self.c))
		if not keep_same_state:
			self.h, self.c = h, c
		return self.pi.compute_policy(h), self.v(h)

	def unchain_backward(self):
		if self.un_init:
			return
		self.h.detach_()
		self.c.detach_()


class A3CMLP(nn.Module, A3CModel):
	def __init__(self, n_actions):
		super(A3CMLP, self).__init__()

		self.head = IndependentHead()
		self.pi = FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
		self.v = FCVFunction(self.head.n_output_channels)
		self.reset_state()

	def reset_state(self):
		return

	def pi_and_v(self, state):
		out = self.head(state)
		return self.pi.compute_policy(out), self.v(out)

	def unchain_backward(self):
		return

class PG(nn.Module, A3CModel):
	def __init__(self, n_actions):
		super(PG, self).__init__()

		self.head = IndependentHead()
		self.pi = FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
		self.reset_state()

	def reset_state(self):
		return

	def pi_and_v(self, state):
		out = self.head(state)
		return self.pi.compute_policy(out)

	def unchain_backward(self):
		return