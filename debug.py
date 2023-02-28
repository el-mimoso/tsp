# import numpy as np
# import pandas as pd
# import random

# data = pd.read_csv('kroA100.tsp', skiprows=[0, 1, 2, 3, 4, 5],
#                    header=None, sep=' ')[:-1]
# data = data.rename(columns={0: "ID", 1: "x", 2: "y"})

# cityList = []
# for i in range(len(data.x.values)):
#     cityList.append([data.x[i], data.y[i]])
# print(cityList)

import sys
import random
import math
import matplotlib.pyplot as plt
import numpy as np

maxint = sys.maxsize
cities = []
order = []
totalCities = 10

population = []

populationSize = 50

fitness = []

bestEver = []

cityPoints = []

recordDistance = sys.maxsize

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

ax1.set_xlim([0, 1000])
ax1.set_ylim([0, 1000])
ax2.set_xlim([0, 1000])
ax2.set_ylim([0, 1000])
plt.ion()
Ln, = ax1.plot(random.sample(range(0, 1000), totalCities),
               marker='o', color='b')
Ln2, = ax2.plot(random.sample(range(0, 1000), totalCities),
                marker='o', color='r')


plt.show()


def plotDistance(order, bestEver):
    cityX = []
    cityY = []
    for i in order:
        cityX.append(cityPoints[i][0])
        cityY.append(cityPoints[i][1])
    npcityX = np.array(cityX)
    npcityY = np.array(cityY)

    if bestEver:
        Ln2.set_ydata(npcityY)
        Ln2.set_xdata(npcityX)
    else:
        Ln.set_ydata(npcityY)
        Ln.set_xdata(npcityX)
    plt.pause(0.001)


def eucDistance(a, b):
    d = math.sqrt((a[1] - b[1]) ** 2 + (a[0] - a[0]) ** 2)
    return d


def createCities():
    for i in range(0, totalCities):
        cityPoints.append((random.sample(range(0, 1000), 2)))
    plt.xlabel("Start:"+str(cityPoints[0]))

    cityPoints
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
            ax1.set_xlabel("Best "+str(d))
            bestEver = population[i].copy()
            plotDistance(bestEver, True)

        fitness.append(1 / (d + 1))
    plotDistance(population[i], False)
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
        newPopulation.append(order)

    population = newPopulation


def crossOver(orderA, orderB):

    start = math.floor(random.randrange(1, len(orderA) - 1))
    end = math.floor(random.randrange(start + 1, len(orderB)))

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
        # print(i)
        if random.random() < mutationRate:
            indexA = math.floor(random.randrange(1, len(order)-1))
            indexB = (indexA + 1) % totalCities
            if(indexB == 0):
                indexB += 1
            print(order)
            swap(order, indexA, indexB)
            print(order)


def swap(a, i, j):
    temp = a[i]
    a[i] = a[j]
    a[j] = temp


if __name__ == "__main__":

    createCities()

    createPopulation()
    a = 1
    while 1:
        calcFitness()
        print(f"Generation {a}")
        nextGeneration()
        a+=1
