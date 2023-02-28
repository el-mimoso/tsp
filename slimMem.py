import sys
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

maxint = sys.maxsize

cities = []
order = []
totalCities = 0

population = []

populationSize = 50

fitness = []

bestEver = []

cityPoints = []

progress = []

recordDistance = sys.maxsize

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

# ax1.set_xlim([0, 100000])
# ax1.set_ylim([0, 100000])
# ax2.set_xlim([0, 100000])
# ax2.set_ylim([0, 100000])
# plt.ion()
# Ln, = ax1.plot(random.sample(range(0, 1000), totalCities),
#                marker='o', color='b')
# Ln2, = ax2.plot(random.sample(range(0, 1000), totalCities),
#                 marker='o', color='r')


plt.show()


# def plotDistance(order, bestEver):
#     cityX = []
#     cityY = []
#     for i in order:
#         cityX.append(cityPoints[i][0])
#         cityY.append(cityPoints[i][1])
#     npcityX = np.array(cityX)
#     npcityY = np.array(cityY)

#     if bestEver:
#         Ln2.set_ydata(npcityY)
#         Ln2.set_xdata(npcityX)
#     else:
#         Ln.set_ydata(npcityY)
#         Ln.set_xdata(npcityX)
#     plt.pause(0.001)


def eucDistance(a, b):
    d = math.sqrt((a[1] - b[1]) ** 2 + (a[0] - a[0]) ** 2)
    return d


def createCities(data):
    for i in range(0, totalCities):
        # n = len(data.x.values)
        # for i in range(len(data.x.values)):
        cityPoints.append([data.x[i], data.y[i]])
    # plt.xlabel("Start:"+str(cityPoints[0]))

    # distance Matrix
    for i in range(0, totalCities):
        temp = []
        for j in range(0, totalCities):
            d = eucDistance(cityPoints[i], cityPoints[j])
            temp.append(d)

        cities.append(temp)
    # print(cities)


def createPopulation():
    for i in range(1, totalCities):
        order.append(i)

    # print ( order )

    for i in range(0, populationSize):
        temp = random.sample(order, len(order))
        temp.insert(0, 0)
        temp.append(0)
        # print ( temp )
        population.append(temp)


def calcDistance(curr_order):
    sum = 0
    for i in range(0, len(curr_order) - 1):
        d = cities[curr_order[i]][curr_order[i + 1]]
        sum += d
    return sum


def calcFitness():
    global recordDistance, bestEver
    currentBest = maxint

    for i in range(0, populationSize):
        d = calcDistance(population[i])

        if (d < recordDistance):
            recordDistance = d
            # ax1.set_xlabel("Best "+str(d))
            bestEver = population[i].copy()
            progress.append(recordDistance)
            # plotDistance(bestEver, True)

        fitness.append(1 / (d + 1))
    # plotDistance(population[i], False)
    print(recordDistance)
    normalizeFitness()


def normalizeFitness():
    sum = 0
    for i in range(0, populationSize):
        sum += fitness[i]
    for i in range(0, populationSize):
        fitness[i] = fitness[i] / sum


def nextGeneration():
    newPopulation = []
    global population
    for i in range(0, populationSize):
        orderA = pickOne(population, fitness)
        orderB = pickOne(population, fitness)
        order = crossOver(orderA, orderB)
        mutate(order, 0.9)
        newOrder = opt2(order)
        if (calcDistance(newOrder) < calcDistance(order)):
            # print("2OPT!")
            newPopulation.append(newOrder)
        else:
            newPopulation.append(order)

    population = newPopulation


def crossOver(orderA, orderB):

    start = math.floor(random.randrange(1, len(orderA) - 1))
    # end = math.floor(random.randrange(start + 1, len(orderB)))
    end = math.floor(random.randrange(1, len(orderB)-1))
    start, end = min(start, end), max(start, end)

    newOrder = orderA[start:end]
    newOrder.insert(0, 0)

    for i in orderB:
        if i not in newOrder:
            newOrder.append(i)
    newOrder.append(0)
    # print ( "start " , start , " " , end ,orderA,orderB,newOrder)
    return newOrder


def pickOne(order, prob):
    index = 0
    r = random.random()
    while r > 0:
        r = r - prob[index]
        index += 1
    index -= 1
    return order[index]


def mutate(order, mutationRate):
    for i in range(0, totalCities):
        if random.random() < mutationRate:
            indexA = math.floor(random.randrange(1, len(order)-1))
            indexB = math.floor(random.randrange(1, len(order)-1))

            # indexB = (indexA + 1) % totalCities
            if(indexB == indexA):
                indexB += 1
            # indexB = math.floor ( random.randrange ( 0 , len ( order ) ) )
            # print ( order , indexA , indexB )
            swap(order, indexA, indexB)


def opt2(order):

    # print(order)
    i, j = random.sample(range(len(order)), 2)
    i, j = min(i, j), max(i, j)
    order = order[:i] + order[i:j+1][::-1] + order[j+1:]
    return order
    # print(order)


def swap(a, i, j):
    # print(a)
    temp = a[i]
    a[i] = a[j]
    a[j] = temp
    return a
    # print(temp)


if __name__ == "__main__":

    data = pd.read_csv('kroA100.tsp', skiprows=[0, 1, 2, 3, 4, 5],
                       header=None, sep=' ')[:-1]
    data = data.rename(columns={0: "ID", 1: "x", 2: "y"})

    totalCities = len(data.values)-1
    createCities(data)

    createPopulation()
    # print(population)
    a = 0

    while a < 10:
        calcFitness()
        print(f"Generation: {a}")
        nextGeneration()
        a += 1
    print(progress)
    print(len(bestEver))
    optRoute = []
    for i in bestEver:
        optRoute.append([cityPoints[i][0],cityPoints[i][1]])
    
    optRoute = np.array(optRoute)   
    print(optRoute)
