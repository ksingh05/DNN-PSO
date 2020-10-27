'''

Author: Karanpreet Singh (Virginia Tech)
Email: kasingh@vt.edu

This code uses trained DNN to do the shape optimization of curvilinearly stiffened panels with two stiffeners

'''

import numpy as np

from pyswarm import pso
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from utils import *

# Define the model architecture 
model = Sequential()
model.add(Dense(90, input_shape=(11,)))
model.add(Activation('tanh'))

for laye in range(1,6):
    model.add(Dense(90))
    model.add(Activation('tanh'))

model.add(Dense(1))
model.add(Activation('linear'))

# Load weights from already trained model
model.load_weights("model.h5")
print("Loaded model from disk")

# Define the input parameters related to panel and applied loading
b           =   0.6096                                 
a           =   0.7112
perimeter   =   2 * (a + b)
shellThick  =   6.25E-03
stiffHeight =   4.00E-02
stiffThick  =   5.43E-03
Nx_Load     =   275.83*1000
Ny_Load     =   496.50*1000
E1          =   73.085E9
nu          =   0.3
Dr          =   E1*shellThick**3/(12*(1-nu**2))


def cost_function(x):
    
    # Return a high cost function as penalty if two stiffeners end points try to come close to
    if (abs(x[0] - x[1]) < 0.3 or
        abs(1 - x[1] + x[0]) < 0.3 or
        abs(x[3] - x[4]) < 0.3 or
        abs(1 - x[4] + x[3])< 0.3 or
        abs(x[0] - x[3]) < 0.02 or
        abs(x[0] - x[4]) < 0.02 or
        abs(x[1] - x[3]) < 0.02 or
        abs(x[1] - x[4]) < 0.02):

        return 1E+07

    # Input feature vector for the DNN
    X = np.concatenate([x,
                       [a / b, 
                        shellThick / b,
                        stiffHeight / shellThick,
                        stiffThick / b,
                        Ny_Load / Nx_Load]])

    # Predict non-dimensionalized buckling parameter
    bucklingParameterNN = DNN_predict(model, X)

    # Evaluate buckling factor
    bucklingfactorNN = bucklingParameterNN * ((np.pi**2) * Dr) / (Nx_Load * b**2)

    # We need to check if the stiffener design variables do not define the stiffener outside the panel
    eA_stiff1   =   x[0] * perimeter
    eB_stiff1   =   x[1] * perimeter
    A_stiff1    =   x[2] * perimeter

    eA_stiff2   =   x[3] * perimeter
    eB_stiff2   =   x[4] * perimeter
    A_stiff2    =   x[5] * perimeter

    x1_stiff1, z1_stiff1, x2_stiff1, z2_stiff1 = stiffener_end_points_coordinates(eA_stiff1, eB_stiff1, a, b)
    x1_stiff2, z1_stiff2, x2_stiff2, z2_stiff2 = stiffener_end_points_coordinates(eA_stiff2, eB_stiff2, a, b)

    alpha_stiff1_x, alpha_stiff1_z = alpha_control_point_coordinates(x1_stiff1, x2_stiff1, z1_stiff1, z2_stiff1, A_stiff1)
    alpha_stiff2_x, alpha_stiff2_z = alpha_control_point_coordinates(x1_stiff2, x2_stiff2, z1_stiff2, z2_stiff2, A_stiff2)

    # Evaluate the penalty value if stiffener is being defined outside the panel
    penalty = calculate_violation(alpha_stiff1_x, alpha_stiff1_z, alpha_stiff2_x, alpha_stiff2_z, a, b)

    costfunc = bucklingfactorNN**(-1) + penalty * 1E+06

    return costfunc

# Lower and upper bound for the optimization
lower_bound = [0.00, 0.25, -0.2, 0.00, 0.25, -0.2]
upper_bound = [0.75, 1.00, 0.2,  0.75, 1.00, 0.2]

print('Running Optimization ....')

# Use pyswarm optimization to conduct Particle Swarm Optimization 
xopt, _ = pso(cost_function, lower_bound, upper_bound, swarmsize=120, maxiter=500,minfunc=5e-4)

print('Shape Optimal Configuration:')
print(xopt)

