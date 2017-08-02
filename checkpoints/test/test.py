import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from  torch.autograd import Variable

class CommnetHead(nn.Module):

	def __init__(self):
		super(CommnetHead, self).__init__()
		self.affine1 = nn.Linear(6,12)
		self.affine2 = nn.Linear(24,6)

	def forward(self, state):
		h1 = F.relu(self.affine1(state))
		c1 = h1.mean(0).expand_as(h1)
		h = torch.cat((h1, c1),1)
		h2 = F.relu(self.affine2(h))

		probs =  F.softmax(h2)
		log_probs = F.log_softmax(h2)

		return -(probs*log_probs).sum(1)

input = np.random.rand(5,6)
input = Variable(torch.from_numpy(input).float(), requires_grad=True)
#print(input)

model = CommnetHead()
a = np.zeros([0,10])
print(a)
print(len(a))
print(a is not None)
print(model)
h = model(input)
print(h)
print(h.size(0))
for i in range(1):
	print(h[i])
	h[i].backward()
	print(input.grad)