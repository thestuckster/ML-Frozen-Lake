import gym
import time
import random

from IPython.display import clear_output

env = gym.make("FrozenLake-v0")
env.render()

n_states = env.observation_space.n
print("number of states: " + str(n_states))

n_actions = env.action_space.n
print("number of actions: " + str(n_actions))

#Define Q learning params
EPSILON = 0.1 # GREEDY POLICY PROPERTY; the agent will do what its told the lower this value.
EXPLORE_EPSILON = .9 #the agent will do more exploring the higher this value
EXPLOIT_EPSILON = 0.01 #very small chance to explore. high chance to exploit for best reward

GAMMA = 0.5 # DISCOUNT FACTOR
ALPHA = 0.1 # LEARNING RATE
QV = 0.0 # R PETRY VALUE (reward)
MAX_QV = 0.0 # MAX CURRENT STATE VALUE

previous_state = 0.0
EPISODE = 0
RUN_EPISODE = 0

# generate random Q_table
Q_TABLE = {}

i = 0
for state in range(n_states):
        for action in range(n_actions):
            i += 1
            Q_TABLE[(state,action)] = round(random.uniform(0,1), 3) #randomly assign a Q value to the table
            
#Epsilon greedy policy
#compare random probability with epsilon at each step 
def greedy_policy(state, EPSILON):
    rand = random.uniform(0,1)  
    if rand < EPSILON:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q_TABLE[(state, x)])


def show_progress():
    print(env.render())
    print('Q_Value: ' + str(QV))
    
    time.sleep(0.5)
    clear_output(wait=True)


#Training operation
print('Training machine...')
env.reset()

#episode loop
for z in range(3000):
    total_reward = 0
    previous_state = env.reset()
    EPISODE += 1
    
    #logic loop
    while True:
        # Reward given to update Q_Value 
        r =  1 #reward value
  
        action = greedy_policy(previous_state, EPSILON) #action to perform according to greedy_policy
        next_state, reward, done, _ = env.step(action) # tell agent to perform action on environment

        # set max q value; search for max value at the next state; must select action available for next state
        MAX_QV = max([Q_TABLE[(next_state, a)] for a in range(n_actions)])

        #update the q value of the previous action taken
        Q_TABLE[(previous_state, action)] = Q_TABLE[(previous_state, action)] + ALPHA *(r + GAMMA  *MAX_QV - Q_TABLE[(previous_state, action)])

        QV = Q_TABLE[(previous_state, action)]

        total_reward += 1
        previous_state = next_state
        
        #end of episode check
        if done:
            if reward == 0:
                # Give agent a bad reward for making a bad action
                r = -100 #reward value;
                action = greedy_policy(previous_state, EPSILON) #action to perform
                next_state, reward, done, _ = env.step(action) #

                # set max q value; search for max value at the next state; must select action available for next state
                MAX_QV = max([Q_TABLE[(next_state, a)] for a in range(n_actions)])

                #update the q value of the previous action taken
                Q_TABLE[(previous_state, action)] = Q_TABLE[(previous_state, action)] + ALPHA *(r + GAMMA  *MAX_QV - Q_TABLE[(previous_state, action)])

                QV = Q_TABLE[(previous_state, action)]
                previous_state = next_state

            if reward == 1:
                # Give agent a good reward for finishing maze
                
                r = 100 #reward value; used to update Q value
                action = greedy_policy(previous_state, EPSILON) #action to perform according to greedy_policy
                next_state, reward, done, _ = env.step(action) # tell agent to perform action on environment

                # set max q value; search for max value at the next state; must select action available for next state
                MAX_QV = max([Q_TABLE[(next_state, a)] for a in range(n_actions)])

                #update the q value of the previous action taken
                Q_TABLE[(previous_state, action)] = Q_TABLE[(previous_state, action)] + ALPHA *(r + GAMMA  *MAX_QV - Q_TABLE[(previous_state, action)])

                QV = Q_TABLE[(previous_state, action)]

                total_reward += 1
                previous_state = next_state
                
            break #quit logic loop
print("Done training!")

#Testing operation
print('real run...')

failures = 0
wins = 0

#episode loop
for z in range(100):
    
    if wins > 0:
            break #quit execution if we win the game
    
    total_reward = 0
    previous_state = env.reset()
    EPISODE += 1
    RUN_EPISODE += 1
    
    start_time = time.time()
    #logic loop
    while True:

        # Reward given to update Q_Value 
        r =  1 #reward value for first step; used to update Q value
  
        action = greedy_policy(previous_state, EPSILON) #action to perform according to greedy_policy
        next_state, reward, done, _ = env.step(action) # tell agent to perform action on environment

        # set max q value; search for max value at the next state; must select action available for next state
        MAX_QV = max([Q_TABLE[(next_state, a)] for a in range(n_actions)])

        #update the q value of the previous action taken
        Q_TABLE[(previous_state, action)] = Q_TABLE[(previous_state, action)] + ALPHA *(r + GAMMA  *MAX_QV - Q_TABLE[(previous_state, action)])

        QV = Q_TABLE[(previous_state, action)]

        total_reward += 1
        previous_state = next_state
        show_progress()
        
        #end of episode check
        if done:
            if reward == 0:
                # Give agent a bad reward for making a bad action
                r = -100
                action = greedy_policy(previous_state, EPSILON)
                next_state, reward, done, _ = env.step(action)

                # set max q value; search for max value at the next state; must select action available for next state
                MAX_QV = max([Q_TABLE[(next_state, a)] for a in range(n_actions)])

                #update the q value of the previous action taken
                Q_TABLE[(previous_state, action)] = Q_TABLE[(previous_state, action)] + ALPHA *(r + GAMMA  *MAX_QV - Q_TABLE[(previous_state, action)])

                QV = Q_TABLE[(previous_state, action)]
                previous_state = next_state
                failures += 1
                print("YOU LOST ):")

            if reward == 1:
                # Give agent a good reward for making a bad action
                r = 100
                action = greedy_policy(previous_state, EPSILON)
                next_state, reward, done, _ = env.step(action)

                # set max q value; search for max value at the next state; must select action available for next state
                MAX_QV = max([Q_TABLE[(next_state, a)] for a in range(n_actions)])

                #update the q value of the previous action taken
                Q_TABLE[(previous_state, action)] = Q_TABLE[(previous_state, action)] + ALPHA *(r + GAMMA  *MAX_QV - Q_TABLE[(previous_state, action)])

                QV = Q_TABLE[(previous_state, action)]

                total_reward += 1
                previous_state = next_state
                wins += 1
                print("YOU WIN!")
                
            print("\nend of episode " + str(EPISODE))
            print("Q_Value for this episode: " + str(QV))
            print("total reward this episode: " + str(total_reward))
            print("running reward total: " + str(total_reward))
            print("\ntotal loses so far: " + str(failures))
            print("total wins so far: " + str(wins))
            
            time.sleep(1)
            clear_output(wait=True)
                
            break #quit logic loop
        

end_time = time.time() - start_time
print("run time: " + time.strftime("%H:%M:%S", time.gmtime(end_time)))
print("total episodes including training: " + str(EPISODE))
print("total episodes for real run: " + str(RUN_EPISODE))
print("total failures for real run: " + str(failures))