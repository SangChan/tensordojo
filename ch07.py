import numpy as np

X = []
Y = []
for i in range(6):
    lst = list(range(i,i+4))
    X.append(list(map(lambda c: [c/10], lst)))
    Y.append((i+4)/10)

X = np.array(X)
Y = np.array(Y)

for i in range(len(X)):
    print(X[i], Y[i])