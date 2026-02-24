import numpy as np
import matplotlib.pyplot as plt

def distance_concentration(dim, n_points=1000):
    """
    Generate n_points random points in 'dim'-dimensional space.
    Compute Euclidean distance of every point from the first point.
    
    Geometric meaning:
    - We create a random high-dimensional cloud.
    - We measure how far other points are from a reference point.
    - We examine how spread out those distances are.
    """
    
    X = np.random.randn(n_points, dim)
    distances = np.linalg.norm(X - X[0], axis=1)
    return distances


dims = [2, 10, 50, 100, 500]

plt.figure()

# Use same bins for all dimensions for fair comparison
bins = 40

for d in dims:
    distances = distance_concentration(d)
    
    plt.hist(
        distances,
        bins=bins,
        alpha=0.4,
        density=True,           # normalize for better comparison
        label=f"{d}D"
    )

plt.title("Distance Concentration in High Dimensions")
plt.xlabel("Distance from Reference Point")
plt.ylabel("Density")
plt.legend(title="Dimension")
plt.show()