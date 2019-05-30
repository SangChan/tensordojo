from tkinter import *

def hello():
    print("hello there")
    

tk = Tk()
btn = Button(tk, text = "click me", command=hello)
btn.pack()

