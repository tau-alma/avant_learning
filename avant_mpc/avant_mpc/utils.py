import numpy as np
from scipy.interpolate import splev

def calculate_curvature(x_s, y_s, distances):
    """
    Calculate the curvature of a 2D curve at each given distance.
    Curvature formula: k = |x'y" - y'x"| / (x'^2 + y'^2)^(3/2)
    """
    # Interpolate x and y at the new distances
    dx = splev(distances, x_s, der=1)
    dy = splev(distances, y_s, der=1)
    ddx = splev(distances, x_s, der=2)
    ddy = splev(distances, y_s, der=2)
    return dx * ddy - dy * ddx / np.power(dx**2 + dy**2, 1.5)

def curvilinear_to_cartesian(state_horizon, tck_x, tck_y):
    N = state_horizon.shape[0]
    xyvb_values = np.zeros((N, 4))  # Assuming structure [x, y, velocity, dot_beta]
    for i in range(N):
        s = state_horizon[i, 0]  # Curvilinear position along the path
        v = state_horizon[i, 1]  # Velocity
        d = state_horizon[i, 2]  # Lateral deviation from the path
        # delta_psi = state_horizon[i, 3]  # Orientation difference relative to path tangent, not used for position calculation
        dot_beta = state_horizon[i, 5]  # Angular velocity

        # Evaluate the reference path splines to get x_ref and y_ref at position s
        x_ref = splev(s, tck_x, der=0)
        y_ref = splev(s, tck_y, der=0)

        # Compute the heading angle (theta) of the path at s using the derivatives of the splines
        dx_ds = splev(s, tck_x, der=1)
        dy_ds = splev(s, tck_y, der=1)
        theta = np.arctan2(dy_ds, dx_ds)

        # Adjust lateral deviation (d) for Cartesian conversion
        # Here we assume d is positive to the right of the path, which is standard in right-handed systems.
        x = x_ref + d * np.cos(theta + np.pi/2)  # Rotate 90 degrees to offset orthogonally to the path
        y = y_ref + d * np.sin(theta + np.pi/2)  # Rotate 90 degrees to offset orthogonally to the path

        xyvb_values[i, :] = [x, y, v, dot_beta]

    return xyvb_values
