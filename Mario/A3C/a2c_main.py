import gym
import numpy as np
from agent_ac2 import Agent


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    alpha = 1e-5
    agent = Agent(alpha=alpha,n_actions=5)

    n_episodes = 10000

    filename = f"a2c_{alpha}.png" #put the hyperparameters in the file name 
    fig_file = "plots/" + filename

    best_score = 0 #start at smallest possible reward

    score_history = []

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    
    for i in range(n_episodes):

        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)

            next_state,reward,done,info = env.step(action)

            score += reward
            if not load_checkpoint:
                agent.learn(state,reward,next_state,done)

            state = next_state
        
        score_history.append(score)
        avg_score = np.mean(score)

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint: #dont override models when testing
                agent.save_models()

    # can plot stuff here

