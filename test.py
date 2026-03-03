import numpy as np
import pysr

def ibp_matrix(d):
    """
    A(d) I = 0
    I = [I1, I2, I3, M1, M2]
    """
    return np.array([
        [1.0, d-2.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, -1.0, 0.0, d-4.0],
        [0.0, 1.0, 1.0, 1.0, 1.0]
    ])

    # equations = np.array([[-x*(4+x), 2+(5-D)*x,      2+x,  -4,     0,   0,       0,     0,    0],
    #                   [       0,  -x*(4+x),        0,   0,    -2, 2+x, (3-D)*x,     0,    0],
    #                   [       0,         0, -x*(4+x),   0,    -2, 2+x, (3-D)*x,     0,    0],
    #                   [       0,         0,        0,  -2, 2-D/2,   0,       0,     0,    0],
    #                   [       0,         0,        0,   0,     1,  -1,       0,     0,    0],
    #                   [       0,         0,        0,   0,    -1,   0,       0, 1-D/2,    0],
    #                  ])

def lhs_sample(n_samples, bounds):
    dim = len(bounds)
    result = np.zeros((n_samples, dim))

    rng = np.random.default_rng()

    for i, (low, high) in enumerate(bounds):
        perm = rng.permutation(n_samples)
        result[:, i] = (perm + rng.random(n_samples)) / n_samples
        result[:, i] = low + result[:, i] * (high - low)

    return result

def is_good_point(d):
    A = ibp_matrix(d)
    s = np.linalg.svd(A, compute_uv=False)

    cond = s.max() / s.min()
    return cond < 1e6

def reduction_map(A, masters, eps=1e-10):
    """
    Compute least-norm reduction:
    I_hard = R * M
    """
    n = A.shape[1]
    hard = [i for i in range(n) if i not in masters]

    Ah = A[:, hard]
    Am = A[:, masters]

    # Moore–Penrose pseudoinverse
    Ah_pinv = np.linalg.pinv(Ah, rcond=eps)

    # I_hard = -Ah^+ Am M
    R = -Ah_pinv @ Am
    return hard, R


def exact1(d, M1, M2):
    c1 = -(d-3)/(2*d-5)
    c2 = -((d-3)*(d-2))/(2*d-5)
    return c1 * M1 + c2 * M2

def exact2(d, M1, M2):
    c1 = -1/(2*d-5)
    c2 = (d-3)/(2*d-5)
    return c1 * M1 + c2 * M2

def exact3(d, M1, M2):
    c1 = -1+1/(2*d-5)
    c2 = -(3*d-8)/(2*d-5)
    return c1 * M1 + c2 * M2


masters = [3, 4]  # M1, M2
hard_index = 0
samples = lhs_sample(1000, [(2.5, 6)])
good_samples = []
for d in samples[:, 0]:
    if is_good_point(d):
        good_samples.append((d))

ds = np.array(good_samples)

print(f"Using {len(ds)} samples")

X = []
y_c1 = []  # coefficient of M1
y_c2 = []  # coefficient of M2

for d in ds:
    A = ibp_matrix(d)
    hard, R = reduction_map(A, masters)

    # first hard integral = hard[0]
    c1, c2 = R[0, :]
    X.append([d])
    y_c1.append(c1)
    y_c2.append(c2)

X = np.array(X)
y_c1 = np.array(y_c1)
y_c2 = np.array(y_c2)

model_c1 = pysr.PySRRegressor(
    maxsize=20,
    niterations=1000,
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    model_selection='best',
)

model_c1.fit(X, y_c1)
print(model_c1)

model_c2 = pysr.PySRRegressor(
    maxsize=20,
    niterations=1000,
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    model_selection='best',
)

model_c2.fit(X, y_c2)
print(model_c2)


def I1_reconstructed(d, M1, M2):
    c1 = model_c1.predict(np.array([[d]]))[0]
    c2 = model_c2.predict(np.array([[d]]))[0]
    return c1 * M1 + c2 * M2


def verify(d):
    M = np.random.randn(2)
    I = np.zeros(5)

    I[masters] = M
    I[hard[0]] = I1_reconstructed(d, M[0], M[1])

    # set remaining hard integrals from numeric reduction
    A = ibp_matrix(d)
    _, R = reduction_map(A, masters)
    I[hard[1:]] = R[1:] @ M

    return np.linalg.norm(A @ I)


def verify2(d):
    M = np.random.randn(2)
    I = np.zeros(5)

    I[masters] = M
    I[hard[0]] = exact1(d, M[0], M[1])
    I[hard[1]] = exact2(d, M[0], M[1])
    I[hard[2]] = exact3(d, M[0], M[1])
    A = ibp_matrix(d)
    print(M)
    print(A)
    print(I)

    return np.linalg.norm(A @ I)

for d in [3.1, 4.3, 5.2]:
    print(d, verify(d))
