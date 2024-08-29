import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random

class Brush:
    def __init__(self, image_size):
        self.x = random.randint(0, image_size[0])
        self.y = random.randint(0, image_size[1])
        self.size = random.randint(1, 10)
        self.color = random.randint(0, 255)
        self.rotation = random.uniform(0, 360)

    def mutate(self):
        self.x += random.randint(-5, 5)
        self.y += random.randint(-5, 5)
        self.size += random.randint(-2, 2)
        self.color += random.randint(-10, 10)
        self.rotation += random.uniform(-15, 15)
        self.size = max(1, min(self.size, 10))
        self.color = max(0, min(self.color, 255))

class Individual:
    def __init__(self, image_size, num_brushes):
        self.image_size = image_size
        self.brushes = [Brush(image_size) for _ in range(num_brushes)]
        self.fitness = 0

    def generate_image(self):
        image = Image.new('L', self.image_size, color=255)
        draw = ImageDraw.Draw(image)
        for brush in self.brushes:
            x1, y1 = brush.x - brush.size, brush.y - brush.size
            x2, y2 = brush.x + brush.size, brush.y + brush.size
            draw.ellipse([x1, y1, x2, y2], fill=brush.color)
        return np.array(image)

class Population:
    def __init__(self, size, image_size, target_image, num_brushes):
        self.size = size
        self.image_size = image_size
        self.target_image = target_image
        self.num_brushes = num_brushes
        self.individuals = [Individual(image_size, num_brushes) for _ in range(size)]

    def fitness_function(self, individual):
        img = individual.generate_image()
        difference = np.abs(img - self.target_image)
        return 1 / (np.mean(difference) + 1)

    def evaluate_fitness(self):
        for individual in self.individuals:
            individual.fitness = self.fitness_function(individual)

    def select_parents(self):
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        return self.individuals[:self.size // 2]

    def crossover(self, parent1, parent2):
        child = Individual(self.image_size, self.num_brushes)
        split = random.randint(0, self.num_brushes)
        child.brushes = parent1.brushes[:split] + parent2.brushes[split:]
        return child

    def mutate(self, individual):
        for brush in individual.brushes:
            if random.random() < 0.1:  # 10% chance of mutation
                brush.mutate()

    def next_generation(self):
        self.evaluate_fitness()
        parents = self.select_parents()
        next_gen = parents.copy()
        
        while len(next_gen) < self.size:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            next_gen.append(child)
        
        self.individuals = next_gen

def run_genetic_algorithm(target_image_path, population_size, num_generations, num_brushes):
    target_image = np.array(Image.open(target_image_path).convert('L').resize((50, 50)))
    image_size = target_image.shape
    
    population = Population(population_size, image_size, target_image, num_brushes)
    
    best_fitnesses = []
    avg_fitnesses = []
    
    for generation in range(num_generations):
        population.next_generation()
        fitnesses = [ind.fitness for ind in population.individuals]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")
            best_individual = max(population.individuals, key=lambda x: x.fitness)
            best_image = best_individual.generate_image()
            plt.imshow(best_image, cmap='gray')
            plt.title(f"Generation {generation}")
            plt.axis('off')
            plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitnesses, label='Best Fitness')
    plt.plot(avg_fitnesses, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.show()

    return population.individuals[0]

# Ejemplo de uso
target_image_path = 'target_image.png'  # Reemplaza con la ruta de tu imagen objetivo
best_individual = run_genetic_algorithm(target_image_path, population_size=50, num_generations=1000, num_brushes=100)

# Mostrar la mejor imagen generada
best_image = best_individual.generate_image()
plt.imshow(best_image, cmap='gray')
plt.title("Best Generated Image")
plt.axis('off')
plt.show()