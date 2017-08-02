import os

for i in range(1000):
	with open(os.path.join('../starcraft', 'winrate'), 'a+') as f:
			f.write('{} {}\n'.format(i+8903, float(i+7187)/(i+8903)))