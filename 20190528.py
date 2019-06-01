import tkinter

def hello():
    print("hello there")


#tk = tkinter.Tk()
#btn = tkinter.Button(tk, text = "click me", command=hello)
#btn.pack()
#tkinter.mainloop()

tk = tkinter.Tk()
canvas = tkinter.Canvas(tk, width = 500, height = 500)
canvas.pack()
canvas.create_line(0,0,500,500)
tkinter.mainloop()