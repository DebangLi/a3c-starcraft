from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl import SoftmaxPolicy

class IndependentHead(nn.Module):

	def __init__(self):
		super(IndependentHead, self).__init__()
		self.affine1 = nn.Linear(68, 256)
		self.affine2 = nn.Linear(256, 512)
		self.affine3 = nn.Linear(512, 128)
		self.n_output_channels = 128

	def forward(self, state):
		h = F.relu(self.affine1(state))
		h = F.relu(self.affine2(h))
		h = F.relu(self.affine3(h))

		return h

class CommnetHead(nn.Module):

	def __init__(self):
		super(CommnetHead, self).__init__()
		self.affine1 = nn.Linear(68, 256)
		self.affine2 = nn.Linear(512,512)
		self.affine3 = nn.Linear(1024,128)
		self.n_output_channels = 128

	def forward(self, state):
		h1 = F.relu(self.affine1(state))
		c1 = h1.mean(0).expand_as(h1)
		h = torch.cat((h1, c1) ,1)
		h2 = F.relu(self.affine2(h))
		c2 = h2.mean(0).expand_as(h2)
		h = torch.cat((h2, c2), 1)
		h3 = F.relu(self.affine3(h))

		return h3


class FCSoftmaxPolicy(nn.Module, SoftmaxPolicy):
	def __init__(self, n_input_channels, n_actions):
		super(FCSoftmaxPolicy, self).__init__()

		self.linear = nn.Linear(n_input_channels, n_actions)

	def forward(self, state):
		return self.linear(state)

	def compute_policy(self, state):
		return self.logits2policy(self(state))

class FCVFunction(nn.Module):
	def __init__(self, n_input_channels):
		super(FCVFunction, self).__init__()

		self.linear = nn.Linear(n_input_channels, 1)

	def forward(self, state):
		return self.linear(state)
