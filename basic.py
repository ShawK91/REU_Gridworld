import numpy as np
from random import randint
import random, sys
import numpy as np
import mod_rqe as mod
import keras

# MACROS
# Gridworld Dimensions
grid_row = 50
grid_col = 50
observability = 1
hidden_nodes = 35
epsilon = 0.5  # Exploration Policy
alpha = 0.5  # Learning rate
gamma = 0.7 # Discount rate
total_steps = 250 #Total roaming steps without goal before termination
num_agents = 1
num_poi = 1
total_train_epoch = 100000
angle_res = 10
online_learning = False
agent_rand = False
poi_rand = False

#ABLATION VARS
use_rnn = False # Use recurrent instead of normal network
success_replay  = True
neat_growth = 2
use_prune = True #Prune duplicates
angled_repr = True


def test_random(gridworld,  illustrate = False):
    rand_suc = 0
    for i in range(1000):
        nn_state, steps, tot_reward = reset_board()
        hist = np.reshape(np.zeros(5 * num_agents), (num_agents, 5))  # Histogram of action taken
        for steps in range(total_steps):  # One training episode till goal is not reached
            for agent_id in range(num_agents):  # 1 turn per agent
                action = randint(0,4)
                hist[agent_id][action] += 1

                # Get Reward and move
                reward, _ = gridworld.move_and_get_reward(agent_id, action)
                tot_reward += reward
                nn_state[agent_id] = gridworld.referesh_state(nn_state[agent_id], agent_id, use_rnn)
                if 0: #illustrate:
                    mod.dispGrid(gridworld)
                    #raw_input('Press Enter')
            if gridworld.check_goal_complete():
                #mod.dispGrid(gridworld)
                break
        #print hist
        if steps < gridworld.optimal_steps * 2:
            rand_suc += 1
    return rand_suc/1000

def test_dqn(q_model, gridworld, illustrate = False, total_samples = 10):
    cumul_rew = 0; cumul_coverage = 0
    for sample in range(total_samples):
        nn_state, steps, tot_reward = reset_board()
        #hist = np.reshape(np.zeros(5 * num_agents), (num_agents, 5))  # Histogram of action taken
        for steps in range(total_steps):  # One training episode till goal is not reached
            for agent_id in range(num_agents):  # 1 turn per agent
                q_vals = get_qvalues(nn_state[agent_id], q_model)  # for first step, calculate q_vals here
                action = np.argmax(q_vals)
                if np.amax(q_vals) - np.amin(q_vals) == 0:  # Random if all choices are same
                    action = randint(0, len(q_model) - 1)
                #hist[agent_id][action] += 1

                # Get Reward and move
                reward, _ = gridworld.move_and_get_reward(agent_id, action)
                tot_reward += reward
                nn_state[agent_id] = gridworld.referesh_state(nn_state[agent_id], agent_id, use_rnn)
                if illustrate:
                    mod.dispGrid(gridworld)
                    print agent_id, action, reward
            if gridworld.check_goal_complete():
                #mod.dispGrid(gridworld)
                break
        cumul_rew += tot_reward/(steps+1); cumul_coverage += sum(gridworld.goal_complete) * 1.0/gridworld.num_poi
    return cumul_rew/total_samples, cumul_coverage/total_samples

def display_q_values(q_model):
    gridworld, agent, nn_state, steps, tot_reward = reset_board(use_rnn)  # Reset board
    for x in range(grid_row):
        for i in range(x):
            _ = mod.move_and_get_reward(gridworld, agent, action=2)
        for y in range(grid_col):
            #mod.dispGrid(gridworld, agent)
            print(get_qvalues(nn_state, q_model))
            _ = mod.move_and_get_reward(gridworld, agent, action=1)
            nn_state = mod.referesh_hist(gridworld, agent, nn_state, use_rnn)
        gridworld, agent, nn_state, steps, tot_reward = reset_board(use_rnn)  # Reset board

def test_nntrain(net, x, y):
    error = []
    for i in range(len(x)):
        input = np.reshape(x[i], (1, len(x[i])))
        error.append((net.predict(input) - y[i])[0][0])
    return error

def reset_board():
    gridworld.reset(agent_rand, poi_rand)
    first_input = []
    for i in range (num_agents):
        first_input.append(gridworld.get_first_state(i, use_rnn))
    return first_input, 0, 0

def get_qvalues(nn_state, q_model):
    values = np.zeros(len(q_model))
    for i in range(len(q_model)):
        values[i] = q_model[i].predict(nn_state)
    return values

def decay(epsilon, alpha):
    if epsilon > 0.0:
        epsilon -= 0.00005
    if alpha > 0.1:
        alpha -= 0.00005
    return epsilon, alpha

def reset_trajectories():
    trajectory_states = []
    trajectory_action = []
    trajectory_reward = []
    trajectory_max_q = []
    trajectory_qval = []
    trajectory_board_pos = []
    return trajectory_states, trajectory_action, trajectory_reward, trajectory_max_q, trajectory_qval, trajectory_board_pos

class statistics():
    def __init__(self):
        self.train_goal_met = 0
        self.reward_matrix = np.zeros(10) + 10000
        self.coverage_matrix = np.zeros(10) + 10000
        self.tr_reward_mat = []
        self.tr_coverage_mat = []
        #self.coverage_matrix = np.zeros(100)
        self.tr_reward = []
        self.tr_coverage = []
        self.tr_reward_mat = []

    def save_csv(self, reward, coverage, train_epoch):
        self.tr_reward.append(reward)
        self.tr_coverage.append(coverage)

        np.savetxt('reward.csv', np.array(self.tr_reward), fmt='%.3f')
        np.savetxt('coverage.csv', np.array(self.tr_coverage), fmt='%.3f')

        self.reward_matrix = np.roll(self.reward_matrix, -1)
        self.reward_matrix[-1] = reward
        self.coverage_matrix = np.roll(self.coverage_matrix, -1)
        self.coverage_matrix[-1] = coverage
        if self.reward_matrix[0] != 10000:
            self.tr_reward_mat.append(np.average(self.reward_matrix))
            np.savetxt('reward_matrix.csv', np.array(self.tr_reward_mat), fmt='%.3f')
            self.tr_coverage_mat.append(np.average(self.coverage_matrix))
            np.savetxt('coverage_matrix.csv', np.array(self.tr_coverage_mat), fmt='%.3f')



if __name__ == "__main__":
    tracker = statistics()
    gridworld = mod.Gridworld(grid_row, grid_col, observability, num_agents, num_poi, agent_rand, poi_rand, angled_repr = angled_repr, angle_res = angle_res) #Create gridworld
    mod.dispGrid(gridworld)

    q_table = np.zeros(gridworld.dim_col * gridworld.dim_row * 5) # Q-table to test learning algorithm
    q_table = np.reshape(q_table, (gridworld.dim_row, gridworld.dim_col, 5))



    for train_epoch in range(total_train_epoch): #Training Epochs Main Loop
        if train_epoch == 0: continue
        epsilon, alpha = decay(epsilon, alpha)
        nn_state, steps, tot_reward = reset_board() #Reset board

        for steps in range(total_steps): #One training episode till goal is not reached
            for agent_id in range(num_agents): #1 turn per agent
                table_pos = [gridworld.agent_pos[agent_id][0] - gridworld.observe, gridworld.agent_pos[agent_id][1] - gridworld.observe]
                prev_table_pos = np.array(table_pos).copy()  # Backup current state as previous state

                q_vals = (q_table[table_pos[0]][[table_pos[1]]])
                action = np.argmax(q_vals[0])
                if np.amax(q_vals[0]) - np.amin(q_vals[0]) == 0:  # Random if all choices are same
                    action = randint(0, 4)

                if random.random() < epsilon and steps % 7 != 0: #Random action epsilon greedy step + data sampling
                    action = randint(0,4)

                #Get Reward and move
                reward, _ = gridworld.move_and_get_reward(agent_id, action)
                tot_reward += reward

                # Update current state
                table_pos = [gridworld.agent_pos[agent_id][0] - gridworld.observe, gridworld.agent_pos[agent_id][1] - gridworld.observe]


                # Get qvalues and maxQ for next iteration
                max_q = np.max(q_table[table_pos[0]][[table_pos[1]]])

                #Online learning
                q_table[prev_table_pos[0]][prev_table_pos[1]][action] = q_vals[0][action] + alpha * (reward - q_vals[0][action] + gamma * max_q)

            if gridworld.check_goal_complete():
                break
        #END OF ONE ROUND OF SIMULATION


        print 'Epochs:', train_epoch, 'Reward', tot_reward, 'Steps: ', steps + 1









