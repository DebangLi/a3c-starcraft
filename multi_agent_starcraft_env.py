import numpy as np

from gym import spaces
from torchcraft_py import proto
import gym_starcraft.utils as utils

import gym_starcraft.envs.starcraft_env as sc

DISTANCE_FACTOR = 16

class MultiAgentEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=3,
                 self_play=False, max_episode_steps=500, name=None):
        super(MultiAgentEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)
        print('Creat a new multi agent env for starcraft!')
        print('IP address is {}'.format(server_ip))
        print('Port is {}'.format(server_port))
        #print(name)
        self.name = name
        self.num_runaway = 0

    def _action_space(self):
        # uid to take action, attack(1) or move(-1), move_degree, move_distance, attacked uid
        action_low = [0, -1.0, -1.0, -1.0, 0]
        action_high = [500, 1.0, 1.0, 1.0, 500]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        # hit points, cooldown, ground range, is enemy, degree, distance (myself)
        # hit points, cooldown, ground range, is enemy (enemy)
        #obs_low = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #obs_high = [100.0, 100.0, 1.0, 1.0, 1.0, 50.0, 100.0, 100.0, 1.0, 1.0]

        # for multi agent, add more observations in the future
        # uid, hit point, shield, colldown, ground range, is enemy, pos.x, pos.y
        obs_low = [0, 0.0, 0.0, 0.0, 0.0, 0.0, -1e6, -1e6]
        obs_high = [500, 100.0, 100.0, 100.0, 1.0, 1.0, 1e6, 1e6]
        return spaces.Box(np.array(obs_low), np.array(obs_high))

    def _make_commands(self, action):
        cmds = []
        if self.state is None or action is None:
            return cmds
        '''
        myself_id = None
        myself = None
        enemy_id = None
        enemy = None
        for uid, ut in self.state['units_myself'].iteritems():
            myself_id = uid
            myself = ut
        for uid, ut in self.state['units_enemy'].iteritems():
            enemy_id = uid
            enemy = ut

        if action[0] > 0:
            # Attack action
            if myself is None or enemy is None:
                return cmds
            # TODO: compute the enemy id based on its position
            cmds.append(proto.concat_cmd(
                proto.commands['command_unit_protected'], myself_id,
                proto.unit_command_types['Attack_Unit'], enemy_id))
        else:
            # Move action
            if myself is None or enemy is None:
                return cmds
            degree = action[1] * 180
            distance = (action[2] + 1) * DISTANCE_FACTOR
            x2, y2 = utils.get_position(degree, distance, myself.x, -myself.y)
            cmds.append(proto.concat_cmd(
                proto.commands['command_unit_protected'], myself_id,
                proto.unit_command_types['Move'], -1, x2, -y2))
        '''
        for i in range(len(action)):
            uid = int(action[i][0])
            attacking = self.state['units_myself'][uid]
            if action[i][1] > 0:
                # Attack action
                if action[i][4] < 0:
                    continue
                attacked_uid = int(action[i][4])
                attacked = self.state['units_enemy'][attacked_uid]
                if attacking is None or attacked is None:
                    print('attacking or attacked is emety! Please check!')
                    continue
                cmds.append(proto.concat_cmd(proto.commands['command_unit_protected'], uid, 
                    proto.unit_command_types['Attack_Unit'], attacked_uid))
            else:
                # Move action
                if attacking is None:
                    print('The unit to move is empty, please chaeck!')
                    continue
                degree = action[i][2] * 180
                distance = (action[i][3] + 1) * DISTANCE_FACTOR
                x2, y2 = utils.get_position(degree, distance, attacking.x, -attacking.y)
                cmds.append(proto.concat_cmd(proto.commands['command_unit_protected'], uid,
                    proto.unit_command_types['Move'], -1, x2, -y2))


        return cmds

    def _make_observation(self):
        myself = None
        enemy = None
        '''
        for uid, ut in self.state['units_myself'].iteritems():
            myself = ut
        for uid, ut in self.state['units_enemy'].iteritems():
            enemy = ut

        obs = np.zeros(self.observation_space.shape)

        if myself is not None and enemy is not None:
            obs[0] = myself.health
            obs[1] = myself.groundCD
            obs[2] = myself.groundRange / DISTANCE_FACTOR - 1
            obs[3] = 0.0
            obs[4] = utils.get_degree(myself.x, -myself.y, enemy.x,
                                      -enemy.y) / 180
            obs[5] = utils.get_distance(myself.x, -myself.y, enemy.x,
                                        -enemy.y) / DISTANCE_FACTOR - 1
            obs[6] = enemy.health
            obs[7] = enemy.groundCD
            obs[8] = enemy.groundRange / DISTANCE_FACTOR - 1
            obs[9] = 1.0
        else:
            obs[9] = 1.0
        '''
        obs = np.zeros([len(self.state['units_myself']) + len(self.state['units_enemy']), self.observation_space.shape[0]])
        n = 0
        # ours
        for uid, ut in self.state['units_myself'].iteritems():
            myself = ut
            obs[n][0] = uid
            obs[n][1] = myself.health
            obs[n][2] = myself.shield
            obs[n][3] = myself.groundCD
            obs[n][4] = myself.groundRange / DISTANCE_FACTOR - 1
            obs[n][5] = 0.0
            obs[n][6] = myself.x
            obs[n][7] = myself.y
            n = n + 1
        for uid, ut in self.state['units_enemy'].iteritems():
            enemy = ut
            obs[n][0] = uid
            obs[n][1] = enemy.health
            obs[n][2] = enemy.shield
            obs[n][3] = enemy.groundCD
            obs[n][4] = enemy.groundRange / DISTANCE_FACTOR - 1
            obs[n][5] = 1.0
            obs[n][6] = enemy.x
            obs[n][7] = enemy.y
	    n = n+1

        return obs

    def _compute_reward(self):
        reward = 0
	'''
        if self.obs[5] + 1 > 1.5:
            reward = -1
        if self.obs_pre[6] > self.obs[6]:
            reward = 15
        if self.obs_pre[0] > self.obs[0]:
            reward = -10
	'''
        if self.obs_pre is not None:
            myself_hp = 0
            enemy_hp = 0
            n_myself = 0
            n_enemy = 0
            myself_hp_pre = 0
            n_myself_pre = 0
            enemy_hp_pre = 0
            n_enemy_pre = 0
            for i in range(len(self.obs)):
                if self.obs[i][5] == 0:
                    myself_hp += self.obs[i][1] + self.obs[i][2]
                    n_myself += 1
                else:
                    enemy_hp += self.obs[i][1] + self.obs[i][2]
                    n_enemy += 1
            myself_hp = myself_hp / (n_myself + np.finfo(np.float32).eps)
            enemy_hp = enemy_hp / (n_enemy + np.finfo(np.float32).eps)

            for j in range(len(self.obs_pre)):
                if self.obs_pre[j][5] == 0:
                    myself_hp_pre += self.obs_pre[j][1] + self.obs_pre[j][2]
                    n_myself_pre += 1
                else:
                    enemy_hp_pre += self.obs_pre[j][1] + self.obs_pre[j][2]
                    n_enemy_pre += 1
            myself_hp_pre = myself_hp_pre / (n_myself_pre + np.finfo(np.float32).eps)
            enemy_hp_pre = enemy_hp_pre / (n_enemy_pre + np.finfo(np.float32).eps)

            reduced_myself = myself_hp_pre - myself_hp
            reduced_enemy = enemy_hp_pre - enemy_hp
            
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
