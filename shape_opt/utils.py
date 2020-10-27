import numpy as np

np.set_printoptions(precision=3)
np.random.seed(632)

def stiffener_end_points_coordinates(eA_stiff, eB_stiff, a, b):
    if eA_stiff <= a:
        x1_stiff = a/2 - eA_stiff
        y1_stiff = 0
    if (eA_stiff > a and eA_stiff <= (a+b)):
        x1_stiff = -a/2
        y1_stiff = eA_stiff - a
    if (eA_stiff > (a+b) and eA_stiff <= (2*a+b)):
        x1_stiff = (eA_stiff - a - b) - a/2
        y1_stiff = b
    if (eA_stiff > (2*a+b) and eA_stiff <= (2*a+2*b)):
        x1_stiff = a/2
        y1_stiff = 2*(a+b) - (eA_stiff)

    if eB_stiff <= a:
        x2_stiff = a/2-eB_stiff
        y2_stiff = 0
    if (eB_stiff > a and eB_stiff <= (a+b)):
        x2_stiff = -a/2
        y2_stiff = eB_stiff - a
    if (eB_stiff > (a+b) and eB_stiff <= (2*a+b)):
        x2_stiff = (eB_stiff - a - b) - a/2
        y2_stiff = b
    if (eB_stiff > (2*a+b) and eB_stiff <= (2*a+2*b)):
        x2_stiff = a/2
        y2_stiff = 2*(a+b) - (eB_stiff)

    return x1_stiff, y1_stiff, x2_stiff, y2_stiff

def DNN_predict(model, X):

    means = [0.25645617,  0.73246508, -0.00702374,  0.25803041,  0.73016091, -0.01095103,  
             1.23864325,  0.01743513,  6.52187407,  0.02236971, 1.02993812]

    std   = [0.16220741, 0.16092781, 0.08576   , 0.16194911, 0.16100401, 0.08547172, 
             0.3806133 , 0.00717618, 2.57792286, 0.01023853, 0.54602738]

    if X[1] < X[0]:
        X[0], X[1] = X[1], X[0]
        X[2] = -1 * X[2]

    if X[4] < X[3]:
        X[3], X[4] = X[4], X[3]
        X[5] = -1 * X[5]

    X -= means
    X /= std

    bucklingParameterNN = model.predict(np.array([X,]))[0][0]

    return bucklingParameterNN

def calculate_violation(alpha_stiff1_x, alpha_stiff1_z, alpha_stiff2_x, alpha_stiff2_z, a, b):

    penalty = 0

    if (alpha_stiff1_x > a/2):
        penalty += abs(alpha_stiff1_x - a/2)

    if alpha_stiff1_x < -a / 2:
        penalty += abs(alpha_stiff1_x + a/2)

    if alpha_stiff1_z < 0:
        penalty += abs(alpha_stiff1_z)

    if alpha_stiff1_z > b:
        penalty += abs(alpha_stiff1_z - b)

    if (alpha_stiff2_x > a/2):
        penalty += abs(alpha_stiff2_x - a/2)

    if alpha_stiff2_x < -a/2.0:
        penalty += abs(alpha_stiff2_x + a/2)

    if alpha_stiff2_z < 0:
        penalty += abs(alpha_stiff2_z)

    if alpha_stiff2_z > b:
        penalty += abs(alpha_stiff2_z - b)

    return penalty
    
def alpha_control_point_coordinates(x1_stiff, x2_stiff, z1_stiff, z2_stiff, A_stiff):

    #http://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point

    tangent_to_stiffener = [x2_stiff- x1_stiff, 0, z2_stiff - z1_stiff]

    a1  =   tangent_to_stiffener[0]
    a2  =   tangent_to_stiffener[1]
    a3  =   tangent_to_stiffener[2]
    b1  =   0
    b2  =   1
    b3  =   0

    normal_to_stiffener  = [a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1]

    stiff_mid_point = [(x2_stiff+x1_stiff)/2, 0, (z2_stiff+z1_stiff)/2]

    alpha_control_point_x = stiff_mid_point[0] + normal_to_stiffener[0] * A_stiff
    alpha_control_point_z = stiff_mid_point[2] + normal_to_stiffener[2] * A_stiff
     
    return alpha_control_point_x, alpha_control_point_z
