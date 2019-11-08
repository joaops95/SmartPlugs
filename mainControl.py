import tkinter as tk
from tkinter import ttk
import time
import threading
import random
import queue
import server
from PIL import ImageTk, Image


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
    
    def show(self):
        self.lift()
 
    def startServer(self, ipadd, port):
        print('server is running')
        s1 = server.Server(ipadd, port)
        conn = s1.connectServer()
        s1.runServer(conn)
        
class Page1(Page):
   def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        ipadd = '127.0.0.1'
        port = '11111'
        label = tk.Label(self, text="Welcome to the SmartHome, click to start server")
        label.pack(side="top", fill="both", expand=True)
        ipaddEntry = tk.Entry(self, textvariable=ipadd)
        ipaddEntry.insert(tk.END, ipadd)
        ipaddEntry.pack()
        portEntry = tk.Entry(self, textvariable=port)
        portEntry.insert(tk.END, port)
        portEntry.pack()
        t1 = threading.Thread(target=self.startServer, args=)
        b1 = tk.Button(self, text="Start the system", command=t1.start(str(ipadd), int(port)))
        b1.pack()
    


class Page2(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label = tk.Label(self, text="This is page 2")
       label.pack(side="top", fill="both", expand=True)

class Page3(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label = tk.Label(self, text="This is page 3")
       label.pack(side="top", fill="both", expand=True)
       
class GuiPart(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)
        p2 = Page2(self)
        p3 = Page3(self)

        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = tk.Button(buttonframe, text="Start the system", command=p1.lift)
        b2 = tk.Button(buttonframe, text="Page 2", command=p2.lift)
        b3 = tk.Button(buttonframe, text="Page 3", command=p3.lift)

        b1.pack(side="left")
        b2.pack(side="left")
        b3.pack(side="left")

        p1.show()
        



if __name__ == "__main__":
    rand = random.Random(  )
    root = tk.Tk()
    root.wm_geometry("400x400")
    gui = GuiPart(root)
    gui.pack(side="top", fill="both", expand=True)
    root.mainloop()