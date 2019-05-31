import tkinter

def hello():
    print("hello there")


tk = tkinter.Tk()
btn = tkinter.Button(tk, text = "click me", command=hello)
btn.pack()
tkinter.mainloop()

