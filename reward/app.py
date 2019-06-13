import json
from sympy import *
from sympy.geometry import *


def lambda_handler(event, context):

    print(event)
    x = Point(0, 0)
    y = Point(1, 1)
    z = Point(2, 2)
    zp = Point(1, 0)
    Point.is_collinear(x, y, z)

    return {
        "value": 1
    }
