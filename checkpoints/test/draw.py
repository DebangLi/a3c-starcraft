import numpy as np 
import matplotlib.pyplot as plt  

winrate_ind = 'winrate_ind'
winrate_a3c = '../starcraft/winrate'
ind_file = open(winrate_ind,'r')
a3c_file = open(winrate_a3c,'r')
ind_lines = ind_file.readlines()
a3c_lines = a3c_file.readlines()
ind_len = len(ind_lines)
a3c_len = len(a3c_lines)

ind_winrate = np.zeros(800)
a3c_winrate = np.zeros(7800)
x = np.zeros(800)
x2 = np.zeros(7800)
i = 0
for line in ind_lines:
	lines = line.split(' ')
	ind_winrate[i] = float(lines[1])
	x[i] = i * 18 /60
	i += 1
	
	if i >= 800:
		break
j = 0
for line in a3c_lines:
	lines = line.split(' ')
	a3c_winrate[j] = float(lines[1])
	x2[j] = j * 1.714 / 60
	j += 1
	if j >= 7800:
		break
	

plt.figure() 
plt.plot(x,ind_winrate,label="Policy Gradient")
plt.plot(x2, a3c_winrate, label='A3C')
plt.xlabel("Train time(min)")  
plt.ylabel("Wining Rate") 
plt.title("PG vs. A3C")
plt.legend() 
plt.savefig("ind.jpg")  