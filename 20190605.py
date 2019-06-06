import tkinter
import random
import time

tk = tkinter.Tk()
tk.title("Game")
tk.resizable(0,0)
tk.wm_attributes("-topmost",1)
canvas = tkinter.Canvas(tk, width=500, height=400, bd = 0, highlightthickness=0)
canvas.pack()
tk.update()