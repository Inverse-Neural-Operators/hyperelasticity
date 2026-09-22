# external imports
import numpy as np

def get_Ibar_normal(I1bar, I2bar32):
    I1bar_normal = I1bar - 3.0
    I2bar32_normal = I2bar32 - 3.0**(3.0/2.0)
    return I1bar_normal, I2bar32_normal

def get_Wbar(model_name, theta):
    if model_name == "NeoHooke":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal
        return Wbar
    elif model_name == "MooneyRivlin":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal + theta[1] * I2bar32_normal
        return Wbar
    elif model_name == "MooneyRivlinQuadratic1":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal**2.0 + theta[1] * I2bar32_normal
        return Wbar
    elif model_name == "MooneyRivlinQuadratic2":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal + theta[1] * I2bar32_normal**2.0
        return Wbar
    elif model_name == "MooneyRivlinQuadratic12":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal**2.0 + theta[1] * I2bar32_normal**2.0 + 0.01 * I1bar_normal
        return Wbar
    elif model_name == "MooneyRivlinLinear1Quadratic1":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal + theta[1] * I1bar_normal**2.0
        return Wbar
    elif model_name == "MooneyRivlinLinear2Quadratic2":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I2bar32_normal + theta[1] * I2bar32_normal**2.0
        return Wbar
    elif model_name == "MooneyRivlinCubic1":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal**3.0 + theta[1] * I2bar32_normal
        return Wbar
    elif model_name == "MooneyRivlinCubic2":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal + theta[1] * I2bar32_normal**3.0
        return Wbar
    elif model_name == "MooneyRivlinCubic12":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal**3.0 + theta[1] * I2bar32_normal**3.0 + 0.01 * I1bar_normal
        return Wbar
    elif model_name == "MooneyRivlinLinear1Cubic1":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal + theta[1] * I1bar_normal**3.0
        return Wbar
    elif model_name == "MooneyRivlinLinear2Cubic2":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I2bar32_normal + theta[1] * I2bar32_normal**3.0
        return Wbar
    elif model_name == "MooneyRivlinQuadratic1Cubic2":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal**2.0 + theta[1] * I2bar32_normal**3.0 + 0.01 * I1bar_normal
        return Wbar
    elif model_name == "MooneyRivlinQuadratic2Cubic1":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal**3.0 + theta[1] * I2bar32_normal**2.0 + 0.01 * I1bar_normal
        return Wbar
    elif model_name == "MooneyRivlinQuadratic1Cubic1":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal**2.0 + theta[1] * I1bar_normal**3.0 + 0.01 * I1bar_normal
        return Wbar
    elif model_name == "MooneyRivlinQuadratic2Cubic2":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I2bar32_normal**2.0 + theta[1] * I2bar32_normal**3.0 + 0.01 * I1bar_normal
        return Wbar
    elif model_name == "Exponential1":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return theta[0] * I1bar_normal**theta[1] + 0.01 * I2bar32_normal
        return Wbar
    elif model_name == "Exponential2":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            return 0.01 * I1bar_normal + theta[0] * I2bar32_normal**theta[1]
        return Wbar
    elif model_name == "TaylorCubic":
        def Wbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            I1bar_max = 3.0 # chosen based on the range of the invariants in the training data
            I2bar32_max = 5.0 # chosen based on the range of the invariants in the training data
            scale = [
                1.0 / I1bar_max,
                1.0 / I2bar32_max,
                1.0 / I1bar_max**2.0,
                1.0 / I2bar32_max**2.0,
                1.0 / I1bar_max**3.0,
                1.0 / I2bar32_max**3.0,
            ]
            # print(scale)
            _Wbar = (theta[0] * scale[0] * I1bar_normal +
                    theta[1] * scale[1] * I2bar32_normal +
                    theta[2] * scale[2] * I1bar_normal**2.0 +
                    theta[3] * scale[3] * I2bar32_normal**2.0 +
                    theta[4] * scale[4] * I1bar_normal**3.0 +
                    theta[5] * scale[5] * I2bar32_normal**3.0)
            return _Wbar
        return Wbar
    else:
        raise ValueError("Model not implemented")

def get_dWbar_dIbar(model_name, theta):
    if model_name == "NeoHooke":
        def dWbar_dIbar(I1bar,I2bar32):
            return np.stack([theta[0] * I1bar**0.0, np.zeros_like(I2bar32)], axis=-1)
        return dWbar_dIbar
    elif model_name == "TaylorCubic":
        def dWbar_dIbar(I1bar,I2bar32):
            I1bar_normal, I2bar32_normal = get_Ibar_normal(I1bar, I2bar32)
            I1bar_max = 3.0 # chosen based on the range of the invariants in the training data
            I2bar32_max = 5.0 # chosen based on the range of the invariants in the training data
            scale = [
                1.0 / I1bar_max,
                1.0 / I2bar32_max,
                1.0 / I1bar_max**2.0,
                1.0 / I2bar32_max**2.0,
                1.0 / I1bar_max**3.0,
                1.0 / I2bar32_max**3.0,
            ]
            # print(scale)
            dWbar_dI1bar = (theta[0] * scale[0] * I1bar_normal**0.0 +
                    2.0 * theta[2] * scale[2] * I1bar_normal**1.0 +
                    3.0 * theta[4] * scale[4] * I1bar_normal**2.0 )
            dWbar_dI2bar32 = (theta[1] * scale[1] * I2bar32_normal**0.0 +
                    2.0 * theta[3] * scale[3] * I2bar32_normal**1.0 +
                    3.0 * theta[5] * scale[5] * I2bar32_normal**2.0)
            return np.stack([dWbar_dI1bar, dWbar_dI2bar32], axis=0)
        return dWbar_dIbar
    else:
        raise ValueError("Model not implemented")



    