import numpy as np
import matplotlib.pyplot as plt

from res.d18v1 import *


initial = np.array(initial)
globalBest = np.array(globalBest)
# print(initial)

title = "d18512"
v= "1"

plt.plot(initial[:, 0], initial[:, 1], marker='.')
# plt.axis('scaled')
plt.title(f"{title} Initial Route")
plt.savefig(f"figures/{title}_{v}_initialRoute.pdf")
plt.close()
# plt.show()

plt.plot(progress)
plt.ylabel('Distance')
plt.xlabel('Generation')
plt.title(f"{title} Progress")
plt.savefig(f"figures/{title}_{v}_progress.pdf")
plt.close()
# plt.show()

plt.plot(globalBest[:, 0], globalBest[:, 1], marker='.')
plt.title(f"{title} Global Best")
plt.savefig(f"figures/{title}_{v}_best.pdf")
plt.close()
# plt.show()
