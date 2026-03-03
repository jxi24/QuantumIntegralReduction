import numpy as np
import pysr
import matplotlib.pyplot as plt
import sympy

masters = [6, 7]

def ibp_matrix(d, x):
    return np.array([
        [-x*(4 + x), 2 + (5 - d)*x, 2 + x, -4, 0, 0, 0, 0],
        [0, -x*(4 + x), 0, 0, -2, 2 + x, (3 - d)*x, 0],
        [0, 0, -x*(4 + x), 0, -2, 2 + x, (3 - d)*x, 0],
        [0, 0, 0, -2, 2 - d/2, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 1 - d/2],
    ], dtype=float)

def reduction_map(A, masters, eps=1e-12):
    n = A.shape[1]
    hard = [i for i in range(n) if i not in masters]

    Ah = A[:, hard]
    Am = A[:, masters]

    Ah_pinv = np.linalg.pinv(Ah, rcond=eps)
    R = -Ah_pinv @ Am

    return hard, R

def lhs_sample(n_samples, bounds, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    dim = len(bounds)
    result = np.zeros((n_samples, dim))
    
    for i, (low, high) in enumerate(bounds):
        perm = rng.permutation(n_samples)
        result[:, i] = (perm + rng.random(n_samples)) / n_samples
        result[:, i] = low + result[:, i] * (high - low)
    
    return result

def is_good_point(d, x, threshold=1e6):
    A = ibp_matrix(d, x)
    s = np.linalg.svd(A, compute_uv=False)
    cond = s.max() / s.min()
    return cond < threshold

def generate_data(samples):
    X = []
    y = []
    
    for d, x in samples:
        if not is_good_point(d, x):
            continue
            
        A = ibp_matrix(d, x)
        hard, R = reduction_map(A, masters)
        
        # coefficient of first master for I1
        i1_index = hard.index(0)
        c1 = R[i1_index, 0]
        
        X.append([d, x])
        # y.append(c1*(x+4)**2)
        y.append(c1)
        
    return np.array(X), np.array(y)

def adaptive_refinement(model, bounds, n_candidates=2000, n_new=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    # dense candidate pool
    candidates = lhs_sample(n_candidates, bounds, rng)
    
    Xc, yc = generate_data(candidates)
    
    if len(Xc) == 0:
        return np.empty((0, 2))
    
    y_pred = model.predict(Xc)
    
    error = np.abs(yc - y_pred)/yc
    
    # pick worst points
    worst_idx = np.argsort(error)[-n_new:]
    worst_points = Xc[worst_idx]
    
    # local Gaussian refinement
    refined = []
    for d, x in worst_points:
        for _ in range(3):
            d_new = d + 0.5 * rng.normal()
            x_new = x + 0.5 * rng.normal()
            
            if bounds[0][0] < d_new < bounds[0][1] and \
               bounds[1][0] < x_new < bounds[1][1]:
                refined.append([d_new, x_new])
    
    return np.array(refined)

bounds = [(1, 10), (-10, 10)]
rng = np.random.default_rng(123)

# initial sample
samples = lhs_sample(2000, bounds, rng)
X, y = generate_data(samples)
    
model = pysr.PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[],
    model_selection='score',
    warm_start=True,
)

for iteration in range(1):
    
    print(f"\nAdaptive iteration {iteration}")
    
    model.fit(X, y)
    print(model)
    print(sympy.denom(model.sympy()))
    
    new_points = adaptive_refinement(model, bounds, rng=rng)
    
    if len(new_points) == 0:
        break
        
    X_new, y_new = generate_data(new_points)
    
    X = np.vstack([X, X_new])
    y = np.concatenate([y, y_new])
    
    print("Dataset size:", len(X))

    # 4. Evaluate residuals
    y_pred = model.predict(X)
    residual = np.abs(y_pred - y)/y

    print("Max residual:", residual.max())
