import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


class City:
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
        # self.adjMatrix = adjMatrix

    def routeDistance(self):
        global adjMatrix
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                # pathDistance += fromCity.distance(toCity)
                pathDistance += adjMatrix[fromCity.index-1, toCity.index-1]
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList):
    route = random.sample(cityList[1:], len(cityList)-1)
    route.insert(0, cityList[0])
    # route = random.sample(cityList, len(cityList))
    return route

def createAdjMatrix(cityList):
    n = len(cityList)
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                adjacency[i,j] = cityList[i].distance(cityList[j])

    return adjacency
            

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual):
    best = individual
    improved = True
    n = len(individual)
    while improved:
            improved = False
            for i in range(1, n-1):
                for j in range(i+1, n-1):
                    if j-i == 1:
                        continue
                    new_individual = individual[:]
                    # this is the 2woptSwap
                    new_individual[i:j] = individual[j-1:i-1:-1]
                    if Fitness(new_individual).routeDistance() < Fitness(best).routeDistance():
                        best = new_individual
                        improved = True
    individual = best
    return best


def mutatePopulation(population):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind])
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children)
    return nextGeneration


def plotRoute(bestRoute, title=""):
    bestRoute.append(bestRoute[0])
    bestRoute = np.array([list([i.x, i.y])for i in bestRoute])

    # plot best initial route
    plt.plot(bestRoute[:, 0], bestRoute[:, 1], marker='o')
    plt.title(title)
    # plt.show()


def geneticAlgorithm(population, popSize, eliteSize, generations):
    pop = initialPopulation(popSize, population)
    progress = []

    distance = (1 / rankRoutes(pop)[0][1])
    progress.append(distance)
    print(f"Initial distance:   {distance}")

    # Best first route
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    # print(bestRoute)
    plotRoute(bestRoute, "Best Initial Route")

    for i in range(0, generations):
        print(f"Current gen {i}")
        pop = nextGeneration(pop, eliteSize)
        progress.append(1 / rankRoutes(pop)[0][1])
        print(f"current best : {1 / rankRoutes(pop)[0][1]}")

    # Best final route.
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    plotRoute(bestRoute, "Best Found Route")

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    # plt.show()

    return bestRoute


dailyRoute = []
dailyRoute.append(City(1000, 500,1))
dailyRoute.append(City(250, 400,2))
dailyRoute.append(City(1000, 600,3))
dailyRoute.append(City(900, 350,4))
dailyRoute.append(City(550, 100,5))
dailyRoute.append(City(800, 500,6))

adjMatrix = createAdjMatrix(dailyRoute)


from timeit import default_timer as timer

tick = timer()

geneticAlgorithm(population=dailyRoute, popSize=10,
                 eliteSize=5, generations=4)
tock = timer()
print(tock-tick)
