import numpy as np



# The BVP objects accepts functions that return arrays that determine the descritization at that point
def makeLaplacian2D(constant=1):
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
        return constant*laplacian/h**2, 1
    return scheme

def makeLaplacian1D(constant=1):
    def scheme(position,ShapeObject):
        h = ShapeObject.h

        #   y -- >
        # x
        # |
        # V
        laplacian = np.array(
            [1, -2, 1]
        )
        # scheme, schemecenter
        return constant*laplacian/h**2, 1
    return scheme



def makeSqareBorderNeumann2D(constant=1):
    def sqareBorderNeumann2D(position,shape):
        # 3 point derivative. O(h^2)
        h = shape.h
        rightGCoeff = -2/h
        #   y -- >
        # x
        # |
        # V

        derivS = np.array([
            [0,   1, 0],
            [0, -4,  2],
            [0,   1, 0],
        ])
        derivSW = np.array([
            [0,   0, 0],
            [0, -4,  2],
            [0,   2, 0],
        ])

        # SørVest
        if position[1] == 0 and position[0] == 0:
            deriv = derivSW
            rightGCoeff = - 4 / h

        # NordVest
        elif position[1] == 1 and position[0] == 0:
            deriv = np.rot90(derivSW,-1)
            rightGCoeff = - 4 / h

        # NordØst
        elif position[1] == 1 and position[0] == 1:
            deriv = np.rot90(derivSW,-2)
            rightGCoeff = - 4 / h
        # SørØst
        elif position[1] == 0 and position[0] == 1:
            deriv = np.rot90(derivSW,-3)
            rightGCoeff =  - 4 / h

        # y==0 er sør
        elif position[1] == 0:
            deriv = derivS
        # x==0  er west
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
        # left, rightGCoeff, rightF, schemecenter
        return constant*deriv/h, rightGCoeff, 1, 1
    return sqareBorderNeumann2D

def makeSqareBorderNeumann1D(constant=1):
    def sqareBorderNeumann1D(position,shape):
        # 3 point derivative. O(h^2)
        h = shape.h
        #   y -- >
        # x
        # |
        # V
        derivW = np.array(
            [0, -2, 2]
          )
        # west
        if position[0] == 0:
            deriv = derivW
        # east
        elif position[0] == 1:
            deriv = np.flip(derivW)
        # else
        else:
            print("Neumann conditions på en kant som ikke er en kant?")
            deriv=0
        # returning coefficients
        # left, rightGCoeff, rightF, schemecenter
        return constant*deriv/h**2, -2/h, 1, 1
    return sqareBorderNeumann1D

def isSqareBorderNeumann(position):
    for coord in position:
        if coord in [0, 1]:
            return True
    return False

