# external imports
import numpy as np
from matplotlib.path import Path

I1bar_min = 3.0
I2bar32_min = 3.0**(3/2)

def sample_Ibar(
        Ibar_sampling_name="grid",
        normalize = True, # returns normalized invariants (I1bar - 3, I2bar32 - 3^(3/2)) if true
        Ibar_range = 3,
        n_Ibar_sampling = 100,
        ):
    I1bar_max = I1bar_min + Ibar_range
    I2bar32_max = I2bar32_min + Ibar_range

    if Ibar_sampling_name == "grid":
        # uniform grid sampling
        x = np.linspace(I1bar_min, I1bar_max, n_Ibar_sampling) # invariant I1bar
        y = np.linspace(I2bar32_min, I2bar32_max, n_Ibar_sampling) # invariant I2bar32
        X, Y = np.meshgrid(x, y)
        X = X - I1bar_min # normalize invariant I1bar - 3
        Y = Y - I2bar32_min # normalize invariant I2bar32 - 3^(3/2)

        # admissible domain
        t = np.linspace(1/20, 20, 500)
        I1bar_normal_curve = t**2 + 2*t**(-1) - I1bar_min # uniaxial tension / equibiaxial tension
        I2bar32_normal_curve = (t**(-2) + 2*t)**(3/2) - I2bar32_min # uniaxial tension / equibiaxial tension
        curve = np.column_stack([I1bar_normal_curve, I2bar32_normal_curve])
        path = Path(curve)
        points = np.column_stack([X.ravel(), Y.ravel()])
        mask = path.contains_points(points).reshape(X.shape)
        X_masked = np.where(mask, X, np.nan)
        Y_masked = np.where(mask, Y, np.nan)
        X_masked = X_masked.reshape(-1)
        Y_masked = Y_masked.reshape(-1)
        I1bar_normal_train = X_masked[~np.isnan(X_masked)]
        I2bar32_normal_train = Y_masked[~np.isnan(Y_masked)]
    
    elif Ibar_sampling_name == "grid_linear":
        # uniform grid sampling
        x = np.linspace(I1bar_min, I1bar_max, n_Ibar_sampling) # invariant I1bar
        y = np.linspace(I2bar32_min, I2bar32_max, n_Ibar_sampling) # invariant I2bar32
        X, Y = np.meshgrid(x, y)
        X = X - I1bar_min # normalize invariant I1bar - 3
        Y = Y - I2bar32_min # normalize invariant I2bar32 - 3^(3/2)

        # admissible domain
        t = np.linspace(1/20, 20, 500)
        I1bar_normal_curve = t**2 + 2*t**(-1) - I1bar_min # uniaxial tension / equibiaxial tension
        I2bar32_normal_curve = (t**(-2) + 2*t)**(3/2) - I2bar32_min # uniaxial tension / equibiaxial tension
        curve = np.column_stack([I1bar_normal_curve, I2bar32_normal_curve])
        path = Path(curve)
        points = np.column_stack([X.ravel(), Y.ravel()])
        mask = path.contains_points(points).reshape(X.shape)
        X_masked = np.where(mask, X, np.nan)
        Y_masked = np.where(mask, Y, np.nan)
        X_masked = X_masked.reshape(-1)
        Y_masked = Y_masked.reshape(-1)
        I1bar_normal_train = X_masked[~np.isnan(X_masked)]
        I2bar32_normal_train = Y_masked[~np.isnan(Y_masked)]

        a = 1.
        b = 3.0
        mask_linear = I2bar32_normal_train <= a * I1bar_normal_train + b
        I1bar_normal_train = I1bar_normal_train[mask_linear]
        I2bar32_normal_train = I2bar32_normal_train[mask_linear]

    # print range of the normalized invariants in the training data
    # print("Minimum I1bar_normal_train:", np.min(I1bar_normal_train))
    # print("Minimum I2bar32_normal_train:", np.min(I2bar32_normal_train))
    # print("Maximum I1bar_normal_train:", np.max(I1bar_normal_train))
    # print("Maximum I2bar32_normal_train:", np.max(I2bar32_normal_train))

    if normalize:
        return I1bar_normal_train, I2bar32_normal_train # normalized invariants (I1bar - 3, I2bar32 - 3^(3/2))
    else:
        return I1bar_normal_train + I1bar_min, I2bar32_normal_train + I2bar32_min # non-normalized invariants (I1bar, I2bar32)
    


