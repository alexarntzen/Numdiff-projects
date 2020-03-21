import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# The BVP objects accepts functions that return arrays that determine the descritization at that point
def makeLaplacian(constant):
    def scheme(position,ShapeObject):
        h = ShapeObject.h

        #   y -- >
        # x
        # |
        # V
        laplacian = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0, 1,  0]
        ])
        # scheme, schemecenter
        return laplacian/h**2, 1
    return scheme



def sqareBorderNeumann(position,BVPobject):
    # 3 point derivative. O(h^2)
    h = BVPobject.h
    #   y -- >
    # x
    # |
    # V
    if position[1] in [0, 1] and position[0] in [0, 1]:
        print("kant:", position)
        return [0], 1, 0, 0

    derivS = np.array([
        [0, 0,    0, 0,    0],
        [0, 0,    0, 0,    0],
        [0, 0, -3/2, 2, -1/2],
        [0, 0,    0, 0,    0],
        [0, 0,    0, 0,    0]
    ])

    # y ==0 er sør
    if position[1] == 0:
        deriv = derivS
    # west
    elif position[0] == 0:
        deriv = np.rot90(derivS,-1)
    # nord
    elif position[1] == 1:
        deriv = np.rot90(derivS,-2)
    # øst
    elif position[0] == 1:
        deriv = np.rot90(derivS,-3)
    else:
        print("Neumann conditions på en kant som ikke er en kant?")
        deriv=0
    # returning coefficients
    # left, rightG, rightF, schemecenter
    return deriv/h, 1, 0, 2

def isSqareBorderNeumann(position):
    if position[1] in [0, 1] and position[0] in [0, 1]:
        return False
    else:
        return True

