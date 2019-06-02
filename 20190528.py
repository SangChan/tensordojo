import tkinter
import random

def hello():
    print("hello there")


#tk = tkinter.Tk()
#btn = tkinter.Button(tk, text = "click me", command=hello)
#btn.pack()
#tkinter.mainloop()

def random_rect(width, height):
    x1 = random.randrange(width)
    y1 = random.randrange(height)
    x2 = x1 + random.randrange(width)
    y2 = y1 + random.randrange(height)
    canvas.create_rectangle(x1,y1,x2,y2)

tk = tkinter.Tk()
canvas = tkinter.Canvas(tk, width = 500, height = 500)
canvas.pack()
#canvas.create_line(0,0,500,500)
for x in range(0,100):
    random_rect(400,400)
tkinter.mainloop()