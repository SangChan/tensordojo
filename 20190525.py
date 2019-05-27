import turtle

t = turtle.Pen()

t.reset()
for x in range(1,9):
    t.forward(100)
    t.left(225)

t.reset()
for x in range(1,38):
    t.forward(100)
    t.left(175)

t.reset()
for x in range(1,20):
    t.forward(100)
    t.left(95)

t. reset()
for x in range(1,19):
    t.forward(100)
    if x % 2 == 0 :
        t.left(175)
    else :
        t.left(225)