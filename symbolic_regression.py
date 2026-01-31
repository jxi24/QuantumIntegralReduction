import pysr
import numpy as np
import matplotlib.pyplot as plt

X = np.random.uniform(size=(10, 6))
Y = np.zeros(10)
for i in range(np.shape(X)[0]):
    sol = np.linalg.solve([[X[i, 0], X[i, 1]],
                           [X[i, 2], X[i, 3]]],
                          [X[i, 4], X[i, 5]])
    Y[i] = sol[0]

print(X)
print(Y)

model = pysr.PySRRegressor(
    maxsize=20,
    niterations=100,
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
)

model.fit(X, Y)
print(model)

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
