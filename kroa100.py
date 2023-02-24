from memetic import *
# Load Data
data = pd.read_csv('kroA100.tsp', skiprows=[0, 1, 2, 3, 4, 5],
                   header=None, sep=' ')[:-1]
data = data.rename(columns={0: "ID", 1: "x", 2: "y"})

# create List and Cost Matrix
cityList = []
for i in range(len(data.x.values)):
    cityList.append(City(data.x[i], data.y[i], int(data.ID[i])))
adjMatrix = createAdjMatrix(cityList)

# initiate
tick = timer()
bi, gB, progress = geneticAlgorithm(
    population=cityList, popSize=10, eliteSize=5, generations=4, adjMat=adjMatrix)
tock = timer()
exeTime = tock-tick
print(f"time: {exeTime}")
# Save to File
saveFile(bestInitial=bi, globalBest=gB, progress=progress,
         timer=exeTime, title="kroaTEst")
