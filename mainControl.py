from tkinter import *
from tkinter import ttk
import time
import threading
import random
import queue
import server
from PIL import ImageTk, Image

class GuiPart:
    def __init__(self, master, queue, endCommand):
        self.queue = queue
        self.master = master
        # Set up the GUI
        tabControl = ttk.Notebook(master)
        tab1 = ttk.Frame(tabControl)            # Create a tab 
        tabControl.add(tab1, text='Tab 1')      # Add the tab
        tabControl.pack(expand=1, fill="both")  # Pack to make visible
        tab2 = ttk.Frame(tabControl)     
        tabControl.add(tab2, text='Tab 1')      # Add the tab
        tabControl.pack(expand=1, fill="both")  # Pack to make visible
        photo = ImageTk.PhotoImage(Image.open('/home/joaos/Desktop/SE/project/assets/incycle.jpg'))
        panel = Label(master=master, image = photo)
        panel.image = photo
        panel.pack()
        welcomeText = Label(master=master, text = "Welcome to the SmartHome v1.0", height=5, width = 40)
        welcomeText.pack()
        b = Button(master, text="Turn on AP", command= self.mainScreen)
        b.pack()

    def mainScreen(self):
        print('mainscreen opened')
        newWindow = Toplevel(master=None)
        Frame(master=newWindow, width=500, height=500).pack()
        welcomeText = Label(master=newWindow, text = "Welcome to MainScreen", height=5, width = 40)
        welcomeText.pack()
        
    def say_hi(self):
        print("turn on ap")
        
        
    def processIncoming(self):
        """Handle all messages currently in the queue, if any."""
        while self.queue.qsize(  ):
            try:
                msg = self.queue.get(0)
                # Check contents of message and do whatever is needed. As a
                # simple test, print it (in real life, you would
                # suitably update the GUI's display in a richer fashion).
                print(msg)
            except queue.Empty:
                # just on general principles, although we don't
                # expect this branch to be taken in this case
                pass

class ThreadedClient:
    """
    Launch the main part of the GUI and the worker thread. periodicCall and
    endApplication could reside in the GUI part, but putting them here
    means that you have all the thread controls in a single place.
    """
    def __init__(self, master):
        """
        Start the GUI and the asynchronous threads. We are in the main
        (original) thread of the application, which will later be used by
        the GUI as well. We spawn a new thread for the worker (I/O).
        """
        self.master = master

        # Create the queue
        self.queue = queue.Queue()

        # Set up the GUI part
        self.gui = GuiPart(master, self.queue, self.endApplication)

        # Set up the thread to do asynchronous I/O
        # More threads can also be created and used, if necessary
        self.running = 1
        self.thread1 = threading.Thread(target=self.workerThread1)
        self.thread2 = threading.Thread(target=self.runServer)
        self.thread1.start()
        self.thread2.start()

        # Start the periodic call in the GUI to check if the queue contains
        # anything
        self.periodicCall(  )

    def periodicCall(self):
        """
        Check every 200 ms if there is something new in the queue.
        """
        self.gui.processIncoming(  )
        if not self.running:
            # This is the brutal stop of the system. You may want to do
            # some cleanup before actually shutting it down.
            import sys
            sys.exit(1)
        self.master.after(200, self.periodicCall)

    def workerThread1(self):
        """
        This is where we handle the asynchronous I/O. For example, it may be
        a 'select(  )'. One important thing to remember is that the thread has
        to yield control pretty regularly, by select or otherwise.
        """
        while self.running:
            # To simulate asynchronous I/O, we create a random number at
            # random intervals. Replace the following two lines with the real
            # thing.
            time.sleep(rand.random(  ) * 1.5)
            msg = rand.random(  )
            self.queue.put(msg)
            
    def runServer(self):
        port = 12312
        s1 = server.Server('10.42.0.1', port)
        s1.connectServer()

    def endApplication(self):
        self.running = 0



if __name__ == "__main__":
    rand = random.Random(  )
    root = Tk()

    client = ThreadedClient(root)
    root.mainloop()