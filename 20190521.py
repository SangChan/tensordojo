import turtle
t = turtle.Pen()

class Animal:
    def __init__(self, species, number_of_legs, color):
        self.species = species
        self.number_of_leg = number_of_legs
        self.color = color

harry = Animal('hippogriff',6,'pink')

import copy
harriet = copy.copy(harry)

print(harry.species)
print(harriet.species)