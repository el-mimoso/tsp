# import numpy as np
# import random
# import operator
# import pandas as pd
# import matplotlib.pyplot as plt


# print("Esta es una prueba :) ")

# # Data for plotting
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)

# title = "some random string"

# plt.plot(t, s)
# plt.ylabel('Y axis')
# plt.xlabel('X axis ')
# plt.savefig(f"figures/{title}_progress.pdf")
# plt.close()
# print("savedd plot!")

import numpy as np


arr = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[8, 9, 0], [4, 5, 6]])

input_dictionary = {'one': 1, 'two': 2}
file = open('Python.txt', 'w')
file.write('Ruta = '+str(arr))
file.write('\n')
file.write(str(arr2))
file.close()
 

#open and read the file after the overwriting:
# f=open("demofile3.txt", "r")
# print(f.read())
