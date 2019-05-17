def testfunc(myname) :
    print("Hello %s" %myname)

print("Sunshine")

another = 100
def variable_test() :
    first = 10
    second = 20
    return first * second * another

print(variable_test())

import time

print(time.asctime())

import sys

print(sys.stdin.readline())

class Things:
    pass

class Inanimate(Things):
    pass

class Animate(Things):
    pass

class Sidewalks(Inanimate):
    pass

class Animals(Animate):
    pass

class Mammals(Animals):
    pass

class Giraffes(Mammals):
    pass