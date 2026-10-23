import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

##############################################################################################################################################################################################
# Objects for outputs
##############################################################################################################################################################################################
class IterCounter:
    """GMRES callback: counts iterations (SciPy doesn't return ITER like MATLAB)
    and, if verbose, prints the iteration number and relative residual ||r||/||f||.
    `step` is the number of iterations per callback: 1 for SciPy, `restart` for
    CuPy (which calls back once per restart cycle)."""
    def __init__(self, verbose=False, step=1):
        self.n = 0
        self.verbose = verbose
        self.step = step
    def __call__(self, rel_res):
        self.n += self.step
        if self.verbose:
            print(f"    iteration {self.n:5d}: relative residual {float(rel_res):.3e}")

##############################################################################################################################################################################################
# Geometry
##############################################################################################################################################################################################
def getCircle(N, center, r=1):
    theta = np.linspace(0, 2.0*np.pi, N, endpoint=False)
    x = center[0] + np.cos(theta)
    y = center[1] + np.sin(theta)
    points = np.vstack((x,y))
    return points

##############################################################################################################################################################################################
# Plotting
##############################################################################################################################################################################################
def rel_error(u_exact, u_approx):
    L_2_error = np.linalg.norm(u_exact-u_approx, ord = 'fro')/np.linalg.norm(u_exact, ord = 'fro')
    return L_2_error

def convTest(mesh_range, num_trials, exact_sol, num_Operator, filepath = 'Images\\convergence_plot.png'):
    meshes = np.floor(np.linspace(mesh_range[0], mesh_range[1], num_trials)).astype(int)
    L_2 = np.zeros(num_trials,)
    for i in range(num_trials):
        print(f"Simulating mesh with {meshes[i]**2} nodes")
        result = num_Operator(meshes[i])
        u_approx = result["solution"]
        xx, yy = result["xgrid"], result["ygrid"]
        u_exact = exact_sol(xx,yy)
        
        L_2[i] = rel_error(u_exact, u_approx) #calculate relative errors for increasingly finer meshes.
    fig3 = plt.figure(3)
    ax3 = plt.axes()
    plt.rcParams['lines.linewidth'] = 3 
    plt.loglog(meshes, 1/meshes, label = r"$\mathcal{O}(\Delta x)$ Reference", color = 'black', linestyle = '--')
    plt.loglog(meshes, 1/meshes**2, label = r"$\mathcal{O}(\Delta x^2)$ Reference", color = 'black')
    plt.loglog(meshes, L_2, label = "Rel Error", color = 'red')
    plt.legend()
    ax3.set_xlabel("Number of FD Nodes")
    ax3.set_ylabel("Relative Error")
    plt.savefig(filepath)
    return 0