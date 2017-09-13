## Switch between independent agent and commnet agent
1. Change in train.py
2. change in rl_helper.py within train_loop function, either agent.act() or agent.act_commnet()
3. Change in test.py within model_eval function
