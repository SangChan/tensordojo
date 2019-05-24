#import turtle
#t = turtle.Pen()

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

import keyword

print(keyword.iskeyword('if'))
print(keyword.iskeyword('ozwald'))
print(keyword.kwlist)

import random

print(random.randint(1,100))
num = random.randint(1,100)

while True:
    print('guess a number between 1 and 100')
    guess = input()
    i = int(guess)
    if i == num :
        print('you guessed right')
        break
    elif i < num :
        print('try higher')
    elif i > num :
        print('try lower')