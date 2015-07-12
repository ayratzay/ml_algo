__author__ = 'Freeman'

import random
import numpy as np
import matplotlib.pyplot as plt
from algo import LinerRegressionNoMatrix
from algo import LinearRegressionMatrix
from algo import LogisticRegressionMatrix

import timeit

start_time = timeit.default_timer()

for i in xrange(100):
    dataset = [(1, a, a + random.gauss(0,1)) for a in range(100)] #(x, y)
    m = LinerRegressionNoMatrix(learning_rate=0.05, iterations=10000)
    m.fit(dataset)
    m.predict(4)

print(timeit.default_timer() - start_time)

#dataset = [(1, a, a ) for a in range(10)] #(x, y)

start_time = timeit.default_timer()

hist = {}
for i in [x/10000.0 for x in range(1,3001) if x % 5 == 0]:
    mm = LinearRegressionMatrix(learning_rate=i, iterations=100)
    dataset = [(1, a, a + random.gauss(0,1)) for a in range(10)] #(x, y)
    dataset_matix = np.array(dataset)
    X = dataset_matix[:, :2]
    y = dataset_matix[:, 2]
    mm.fit(X, y)
#mm.predict(np.array([1, 4]))

lr, it, mse = [],[],[]
for key,value in hist.iteritems():
    lr.append(key)
    it.append(value[0])
    mse.append(value[1])

lr.sort(), mse.sort(), it.sort()

plt.plot(lr, mse, 'b-')
plt.plot(lr, it, 'r--')


X, y
mdl = LogisticRegressionMatrix(iterations=1000, learning_rate=0.05)
mdl.fit(X, y)
mdl.predict(X)