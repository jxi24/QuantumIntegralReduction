# import pysr
import numpy as np
import matplotlib.pyplot as plt

"""
                     I22       I12         I21   I03   I02  I20    I11      I10 
    equations = [[-x*(4+x), 2+(5-D)*x,      2+x,  -4,     0,   0,       0,     0,    0],
                 [       0,  -x*(4+x),        0,   0,    -2, 2+x, (3-D)*x,     0,    0],
                 [       0,         0, -x*(4+x),   0,    -2, 2+x, (3-D)*x,     0,    0],
                 [       0,         0,        0,  -2, 2-D/2,   0,       0,     0,    0],
                 [       0,         0,        0,   0,     1,  -1,       0,     0,    0],
                 [       0,         0,        0,   0,    -1,   0,       0, 1-D/2,    0],
                ]
"""

x=1
D=4
equations = np.array([[-x*(4+x), 2+(5-D)*x,      2+x,  -4,     0,   0,       0,     0],
                      [       0,  -x*(4+x),        0,   0,    -2, 2+x, (3-D)*x,     0],
                      [       0,         0, -x*(4+x),   0,    -2, 2+x, (3-D)*x,     0],
                      [       0,         0,        0,  -2, 2-D/2,   0,       0,     0],
                      [       0,         0,        0,   0,     1,  -1,       0,     0],
                      [       0,         0,        0,   0,    -1,   0,       0, 1-D/2],
                     ])

print(equations)
print(equations[:, 0:-2])
print(equations[:, -2:])

r = -np.linalg.pinv(equations[:, 0:-2], rcond=1e-8) @ equations[:, -2:]
print(r @ np.transpose(equations[:, -2:]))
print((r @ np.transpose(equations[:, -2:]))[0])

# sol = np.linalg.solve(equations[:,:-1], equations[:, -1])
# print(sol)


# X = np.random.uniform(size=(10, 6))
# Y = np.zeros(10)
# for i in range(np.shape(X)[0]):
#     sol = np.linalg.solve([[X[i, 0], X[i, 1]],
#                            [X[i, 2], X[i, 3]]],
#                           [X[i, 4], X[i, 5]])
#     Y[i] = sol[0]
# 
# print(X)
# print(Y)
# 
# model = pysr.PySRRegressor(
#     maxsize=20,
#     niterations=100,
#     elementwise_loss="loss(prediction, target) = (prediction - target)^2",
# )
# 
# model.fit(X, Y)
# print(model)

# Z = model.predict(X)
# 
# bins = np.linspace(0, 1, 51)
# plt.hist(X[:, 0], weights=Y, bins=bins)
# plt.hist(X[:, 0], weights=Z, bins=bins)
# plt.savefig("tmp0.png")
# plt.close()
# 
# plt.hist(X[:, 1], weights=Y, bins=bins)
# plt.hist(X[:, 1], weights=Z, bins=bins)
# plt.savefig("tmp1.png")
# plt.close()
# 
# plt.hist(X[:, 2], weights=Y, bins=bins)
# plt.hist(X[:, 2], weights=Z, bins=bins)
# plt.savefig("tmp2.png")
# plt.close()
# 
# plt.hist(X[:, 3], weights=Y, bins=bins)
# plt.hist(X[:, 3], weights=Z, bins=bins)
# plt.savefig("tmp3.png")
# plt.close()
# 
# plt.hist(X[:, 4], weights=Y, bins=bins)
# plt.hist(X[:, 4], weights=Z, bins=bins)
# plt.savefig("tmp4.png")
# plt.close()
