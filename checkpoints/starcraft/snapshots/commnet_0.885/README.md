## The reward setting


reward = 0
            if reduced_enemy > reduced_myself:
                reward = (reduced_enemy - reduced_myself) * 2
                self.num_runaway = 0
            elif reduced_enemy <= reduced_myself and reduced_enemy > 0:
                self.num_runaway = 0
                reward = (reduced_enemy - reduced_myself) 
            elif reduced_enemy == 0 and reduced_myself > 0:
                self.num_runaway = 0
                reward = (reduced_enemy - reduced_myself)*2
            elif reduced_myself == 0 and reduced_enemy == 0:
                self.num_runaway += 1
                reward = -20
            #reward -= 0.1 * self.num_runaway
            
            #reward = reduced_enemy - reduced_myself
            #print(reward)

            #if reduced_enemy == 0:
            #    reward = -30   
        #print(reward)
        if self._check_done() and not bool(self.state['battle_won']):
            #print('******************Battle Failed!!!**********************')
            reward = -50
        if self._check_done() and bool(self.state['battle_won']):
            #print('*******************Battle Won!!!*************************')
            reward = 100
            self.episode_wins += 1
        if self.episode_steps >= self.max_episode_steps:
            #print('******************THE END!*******************************')
            reward = -100
        return reward
