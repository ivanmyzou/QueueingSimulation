#Queueing Simulations
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk

import util


class QS(tk.Frame):

    def __init__(self, master=None):
        ttk.Frame.__init__(self, master)
        self.pack()
        self.create()

    def create(self):
        ttk.Label(self, text = '').grid(row=0, column=0) #padding
        self.tag1 = ttk.Label(self, text = 'Interarrival Times', width = 25,
                              anchor = 'nw', justify = 'left', font = ('Courier New', 12))
        self.tag1.grid(row=1, column=0)
        self.dropdown1 = ttk.Combobox(self, width = 15, textvariable = tk.StringVar(), state = "readonly")
        self.dropdown1['values'] = ['M (Markovian)', 'G (General)']
        self.dropdown1.current(0) #defaults to M
        self.dropdown1.grid(row=1, column=1)

        ttk.Label(self, text='').grid(row=2, column=0)  # padding
        self.tag2 = ttk.Label(self, text = 'Service Workloads', width = 25,
                              anchor = 'nw', justify = 'left', font = ('Courier New', 12))
        self.tag2.grid(row=3, column=0)
        self.dropdown2 = ttk.Combobox(self, width = 15, textvariable = tk.StringVar(), state = "readonly")
        self.dropdown2['values'] = ['M (Markovian)', 'G (General)']
        self.dropdown2.current(0)  # defaults to M
        self.dropdown2.grid(row=3, column=1)

        ttk.Label(self, text='').grid(row=4, column=0)  # padding
        self.tag3 = ttk.Label(self, text = 'Number of Servers', width = 25,
                              anchor = 'nw', justify = 'left', font = ('Courier New', 12))
        self.tag3.grid(row=5, column=0)
        self.nserver_entry = ttk.Entry(self, width = 10)
        self.nserver_entry.insert(0, '1')
        self.nserver_entry.grid(row=5, column=1)



if __name__ == '__main__':
    root = ThemedTk(theme="breeze")
    root.title(string='Queuing Simulations')
    root.geometry('800x600')
    app = QS(master=root)
    app.mainloop()
