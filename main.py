import random
import numpy as np

class Individual:
    def __init__(self, chromosome_length):
        self.chromosome = np.random.randint(2, size=chromosome_length)
        self.fitness = 0

    def calculate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.chromosome)

    def mutate(self, mutation_rate):
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                self.chromosome[i] = 1 if self.chromosome[i] == 0 else 0

    def crossover(parent1, parent2):
        return parent1, parent2


class Population(Individual):
    def __init__(self, population_size, chromosome_length, mutation_rate=0.01):
        self.individuals = [Individual(chromosome_length) for _ in range(population_size)]
        self.mutation_rate = mutation_rate

    def check_fitness(self):
        for individual in self.individuals:
            individual.calculate_fitness(self.fitness_function)

    def get_parent(self):
        total_fitness = sum(individual.fitness for individual in self.individuals)
        probabilities = [individual.fitness / total_fitness for individual in self.individuals]
        return self.individuals[np.random.choice(len(self.individuals), p=probabilities)]

    def next_generation(self):
        new_population = []
        self.check_fitness()

        for _ in range(len(self.individuals) // 2):
            parent1 = self.get_parent()
            parent2 = self.get_parent()

            child1, child2 = self.crossover(parent1, parent2)

            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)

            new_population.extend([child1, child2])

        self.individuals = new_population

    def get_best_fitness(self):
        return max(self.individuals, key=lambda indiv: indiv.fitness)

# Algorithm Params
population_size = 100
generations = 1000
chromosome_length = 10
mutation_rate = 0.01
best_individual = None

# Init population object
population = Population(population_size, chromosome_length, mutation_rate)

# Ejecutar el algoritmo genético
for gen in range(generations):
    population.next_generation()
    best_individual = population.get_best_fitness()
    print(f"Generación {gen + 1}: Mejor fitness = {best_individual.fitness}")

print(f"La mejor solución encontrada es: {best_individual.chromosome}")
