import visdom
import numpy as np 
import time

clear = False

vis = visdom.Visdom(env='Starcraft')

command_file = 'command_id'

while True:
	file = open(command_file, 'r')
	flines = file.readlines()
	length = len(flines)

	#commands = np.zeros(length)
	commands = np.zeros([length,2])
	i = 0
	for line in flines:
		commands[i][0] = i
		commands[i][1] = np.float(line)
		i += 1
	vis = visdom.Visdom(env='Starcraft')
	vis.scatter(commands[length-100000: length-1],opts=dict(markersize=1), win=0)
	print('length: {}'.format(length))
	time.sleep(10)
	