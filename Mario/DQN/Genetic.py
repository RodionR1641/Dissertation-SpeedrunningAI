import random
import numpy as np

from Mario.DQN.Rainbow.agent import Agent
from Mario.DQN.mario import DQN_Mario

# using a genetic algorithm to see what 
# also checking CNN vs ViT here as a bonus
class GeneticHyperparameterSearch:
    def __init__(self,env,pop_size=10,generations=5,mutation_rate=0.1):
        self.env = env
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    #initialise populations with different hyper parameters
    def init_population(self):
        population = []
        for _ in range(self.pop_size):
            agent_param = {
                "learn_rate": 10 ** random.uniform(-5,-3), #between 1e-5 and 1e-3
                "gamma" : random.uniform(0.9,0.99),
                "sync_network_rate": random.randint(10_000,50_000)
            }
            population.append(agent_param)
        return population

    def train_agent(self,params,input_dims,device="cpu"):
        agent = Agent(
            input_dims=input_dims,  # Assuming input is 4 stacked frames of 84x84
            device=device,
            epsilon=1.0,
            min_epsilon=0.1,
            nb_warmup=250_000,
            nb_actions=5,
            memory_capacity=100_000,
            batch_size=32,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            sync_network_rate=params["sync_network_rate"],
            use_vit=False
        )

        # Train for a fixed number of episodes and return performance
        average_losses = []
        for episode in range(1):  # Train for 5 episodes for average(keep it short)
            stats = agent.train(env=self.env, epochs=5000)  # Train for "small" amount of epochs
            average_losses.append(stats["AverageLoss"]) #average loss will be over last 100, but it is representative to "final" trained agent

        return np.mean(average_losses)
    
    #select top k best performing agents
    def select_best_agents(self,population,fitness_scores,top_k=4):
        sorted_indices = np.argsort(fitness_scores) #ascending order of fitness score indices
        #best_agents are the ones with lowest average loss
        best_agents = []
        for i in range(top_k):
            index = sorted_indices[i]
            best_agents.append(population[index])
        return best_agents

    def crossover(self,parent1,parent2):
        child = {
            "learning_rate": random.choice([parent1["learning_rate"], parent2["learning_rate"]]),
            "gamma": random.choice([parent1["gamma"], parent2["gamma"]]),
            "sync_network_rate": random.choice([parent1["sync_network_rate"], parent2["sync_network_rate"]]),
            "use_vit": random.choice([parent1["use_vit"], parent2["use_vit"]])
        }
        return child

    def mutate(self, agent):
        "Slightly Mutates a child's hyperparameters with a small probability."
        if random.random() < self.mutation_rate:
            agent["learning_rate"] *= random.choice([0.8, 1.2])  # Slightly change LR
        if random.random() < self.mutation_rate:
            agent["gamma"] += random.uniform(-0.02, 0.02)  # Small change to gamma
        if random.random() < self.mutation_rate:
            agent["sync_network_rate"] = random.randint(1000, 50000)  # Re-randomize
        if random.random() < self.mutation_rate:
            agent["use_vit"] = not agent["use_vit"]  # Flip CNN/ViT

        return agent
    
    def run_evolution(self,input_dims,device="cpu"):
        population = self.init_population() #list of parameters

        for gen in range(self.generations):
            print(f"\n---- Generation {gen+1} ----")

            fitness_scores = []

            for agent_params in population:
                score = self.train_agent(params=agent_params,input_dims=input_dims,device=device)
                fitness_scores.append(score) #the index here will be the same as corresponding population index
            
            best_agents = self.select_best_agents(population,fitness_scores)

            new_population = []
            while len(new_population) < self.pop_size:
                parent1,parent2 = random.sample(best_agents,2)
                child = self.crossover(parent1,parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        final_fitness = [self.train_agent(agent) for agent in population]
        best_final_agent = population[np.argmax(final_fitness)]
        return best_final_agent #best parameters
