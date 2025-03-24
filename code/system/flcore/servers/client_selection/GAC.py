import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GAClientSelection:
    def __init__(self, num_clients, num_join_clients, max_poisoned_ratio=0.4, population_size=50, generations=100):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.max_poisoned_ratio = max_poisoned_ratio
        self.population_size = population_size
        self.generations = generations

        self.population = self.initialize_population()
        self.performance_history = np.random.rand(num_clients)  # Example of client performance
        self.client_features = np.random.randn(num_clients, 10)  # Random features for similarity calculation

        # Debug information
        self.numbers_of_selections = [0] * self.num_clients
        self.fitness_history = []

    def initialize_population(self):
        """ Initialize the population with random binary chromosomes """
        population = []
        for _ in range(self.population_size):
            chromosome = np.zeros(self.num_clients)
            selected_indices = random.sample(range(self.num_clients), self.num_join_clients)
            chromosome[selected_indices] = 1
            population.append(chromosome)
        return population

    def fitness_function(self, chromosome):
        """ Fitness function evaluating accuracy and diversity """
        selected_clients = np.where(chromosome == 1)[0]
        if len(selected_clients) != self.num_join_clients:
            return -np.inf  # Invalid selection, we need exact num_join_clients selected

        # Calculate performance for selected clients
        performance = np.mean(self.performance_history[selected_clients])

        # Calculate diversity (using cosine similarity between clients' features)
        feature_subset = self.client_features[selected_clients]
        similarity_matrix = cosine_similarity(feature_subset)
        diversity = np.mean(similarity_matrix)

        # Penalize for selecting poisoned clients (low performance)
        poisoned_penalty = self.detect_poisoned_clients(selected_clients)

        # Fitness combines performance, diversity and penalizes poisoned clients
        fitness = performance - poisoned_penalty + diversity
        return fitness

    def detect_poisoned_clients(self, selected_clients):
        """ Simple heuristic to detect poisoned clients based on performance anomaly """
        penalty = 0
        for client in selected_clients:
            if self.performance_history[client] < 0.3:  # Threshold for low performance
                penalty += 1
        return penalty

    def crossover(self, parent1, parent2):
        """ Single point crossover between two parents """
        crossover_point = random.randint(1, self.num_clients - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, chromosome):
        """ Mutation: flip a random bit in the chromosome """
        mutation_point = random.randint(0, self.num_clients - 1)
        chromosome[mutation_point] = 1 - chromosome[mutation_point]
        return chromosome

    def select(self):
        """ Select the fittest individuals from the population """
        fitness_scores = [self.fitness_function(chromosome) for chromosome in self.population]
        sorted_indices = np.argsort(fitness_scores)[::-1]
        self.fitness_history.append(max(fitness_scores))  # Track best fitness
        return [self.population[i] for i in sorted_indices[:self.population_size // 2]]

    def run(self):
        """ Run the genetic algorithm """
        for generation in range(self.generations):
            #print(f"Generation {generation + 1}")
            #print("-" * 30)

            selected_parents = self.select()
            new_population = []

            # 如果只有一個父母，複製其染色體以保持族群大小
            if len(selected_parents) < 2:
                #print("Insufficient parents for crossover. Duplicating best parent.")
                selected_parents = selected_parents * 2  # 複製使其可進行交配

            for i in range(0, len(selected_parents) - 1, 2):
                parent1, parent2 = selected_parents[i], selected_parents[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))

            # 如果是奇數，將最後一個父母直接加入族群
            if len(selected_parents) % 2 == 1:
                last_parent = selected_parents[-1]
                new_population.append(self.mutate(last_parent))

            self.population = new_population

            best_chromosome = self.population[0]
            best_fitness = self.fitness_function(best_chromosome)

            # Debug: Print fitness and selection details
            #print(f"Best fitness this generation: {best_fitness:.4f}")
            selected_clients = np.where(best_chromosome == 1)[0]
            #print(f"Selected clients: {selected_clients}")
            #print(f"Fitness history: {self.fitness_history}")

            # Update selection counts for debugging
            for client in selected_clients:
                self.numbers_of_selections[client] += 1

            #print(f"Numbers of selections: {self.numbers_of_selections}")
            #print(f"Performance history: {self.performance_history}")
            #print("=" * 30)

        # Return the best client selection from the final population
        best_selection = np.where(self.population[0] == 1)[0]
        return best_selection

    def select_clients(self, epoch):
        """ Wrapper for compatibility with external calls """
        print(f"Selecting clients for epoch {epoch}")
        return self.run()

    def update(self, selected_clients, rewards):
        reward_decay = 1
        for client, reward in zip(selected_clients, rewards):
            self.numbers_of_selections[client] += 1
            self.performance_history[client] = (
                self.performance_history[client] * reward_decay + reward
            )

        #print("Updated numbers of selections:", self.numbers_of_selections)
        #print("Updated performance history:", self.performance_history)
