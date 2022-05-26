#Queueing Simulations

import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import tkinter.font

from distribution import distributions

import util

class QS(tk.Frame):

    def __init__(self, master=None):
        self.master = master
        ttk.Frame.__init__(self, master)
        self.pack()
        self.create()

    def restart(self):
        self.destroy() #remove all labels and create again
        master = self.master
        ttk.Frame.__init__(self, master)
        self.pack()
        self.create()

    def create(self):
        ttk.Label(self, text = ' \n \n ').grid(row=0, column=0) #padding
        ttk.Button(self, text='Restart', command = self.restart).grid(row=0, column=1)
        ttk.Label(self, text = 'Distributions', font = tkinter.font.Font(family='Helvetica', size=10, weight='bold')).grid(row=0, column=3)
        ttk.Label(self, text = '  Parameters', font = tkinter.font.Font(family='Helvetica', size=10, weight='bold'),
                  anchor="center").grid(row=0, column=5, columnspan=3)

        self.tag1 = ttk.Label(self, text = 'Interarrival Times', width = 25,
                              anchor = 'nw', justify = 'left', font = ('Courier New', 12))
        self.tag1.grid(row=1, column=0)
        self.dropdown1 = ttk.Combobox(self, width = 15, textvariable = tk.StringVar(), state = "readonly")
        self.dropdown1['values'] = ['M (Markovian)', 'G (General)']
        self.dropdown1.current(0) #defaults to M
        self.dropdown1.grid(row=1, column=1)
        self.dropdown1.bind("<<ComboboxSelected>>", lambda event: self.selectMG_1(event)) #disable/enable distribution selections
        ttk.Label(self, text = '   ').grid(row=1, column=2)  # padding
        self.dis1 = ttk.Combobox(self, width = 15, textvariable = tk.StringVar(), state = "disabled")
        self.dis1['values'] = distributions
        self.dis1.current(0) #defaults to M
        self.dis1.grid(row=1, column=3)
        ttk.Label(self, text = '  ').grid(row=1, column=4)  # padding
        self.para_1_1 = ttk.Entry(self, width=8)
        self.para_1_1.grid(row=1, column=5)
        ttk.Label(self, text=' ').grid(row=1, column=6)  # padding
        self.para_1_2 = ttk.Entry(self, width=8)
        self.para_1_2.grid(row=1, column=7)
        ttk.Label(self, text='      ').grid(row=1, column=8)  # padding
        self.a_add = ttk.Button(self, text = 'add to list')
        self.a_add.grid(row=1, column=9)

        ttk.Label(self, text='').grid(row=2, column=0)  # padding
        self.tag2 = ttk.Label(self, text = 'Service Workloads', width = 25,
                              anchor = 'nw', justify = 'left', font = ('Courier New', 12))
        self.tag2.grid(row=3, column=0)
        self.dropdown2 = ttk.Combobox(self, width = 15, textvariable = tk.StringVar(), state = "readonly")
        self.dropdown2['values'] = ['M (Markovian)', 'G (General)']
        self.dropdown2.current(0)  # defaults to M
        self.dropdown2.grid(row=3, column=1)
        self.dropdown2.bind("<<ComboboxSelected>>", lambda event: self.selectMG_2(event)) #disable/enable distribution selections
        ttk.Label(self, text = '   ').grid(row=3, column=2)  # padding
        self.dis2 = ttk.Combobox(self, width = 15, textvariable = tk.StringVar(), state = "disabled")
        self.dis2['values'] = distributions
        self.dis2.current(0) #defaults to M
        self.dis2.grid(row=3, column=3)
        ttk.Label(self, text='  ').grid(row=3, column=4)  # padding
        self.para_2_1 = ttk.Entry(self, width=8)
        self.para_2_1.grid(row=3, column=5)
        ttk.Label(self, text=' ').grid(row=3, column=6)  # padding
        self.para_2_2 = ttk.Entry(self, width=8)
        self.para_2_2.grid(row=3, column=7)
        ttk.Label(self, text='      ').grid(row=3, column=8)  # padding
        self.s_add = ttk.Button(self, text = 'add to list')
        self.s_add.grid(row=3, column=9)

        ttk.Label(self, text='').grid(row=4, column=0)  # padding
        self.tag3 = ttk.Label(self, text = 'Number of Servers', width = 25,
                              anchor = 'nw', justify = 'left', font = ('Courier New', 12))
        self.tag3.grid(row=5, column=0)
        self.nserver_entry = ttk.Entry(self, width = 5)
        self.nserver_entry.insert(0, '1')
        self.nserver_entry.grid(row=5, column=1)
        self.pr = ttk.Label(self, text = '  processing rate (\u03bc): ')
        self.pr.grid(row=5, column=2, columnspan=2)  # padding
        self.pr_entry = ttk.Entry(self, width = 5)
        self.pr_entry.insert(0, '1')
        self.pr_entry.grid(row=5, column=4, columnspan=2)
        self.pr_add = ttk.Button(self, text='add to list')
        self.pr_add.grid(row=5, column=9)

        ttk.Label(self, text='').grid(row=6, column=0)  # padding
        self.simple = ttk.Button(self, text = 'Simple Scenarios \u2191', width = 25)
        self.simple.grid(row=7, column=0)
        ttk.Label(self, text='-' * 25).grid(row=8, column=0)  # padding
        self.simple = ttk.Button(self, text = 'Advanced Scenarios \u2193', width = 25)
        self.simple.grid(row=9, column=0)
        ttk.Label(self, text='').grid(row=10, column=0)  # padding

        self.display = tk.Label(self, text = 'Simple Scenario Mode', width = 40, font = ("Arial", 14, 'bold'), fg = '#143D3D',
                                bg = '#f8feff').grid(row=7, column=1, columnspan=9, rowspan=3)

        #advanced scenarios
        self.advanced = ttk.Frame(self)
        self.advanced.grid(row=11, column=0, columnspan=10)
        tk.Label(self.advanced, text=' Order \n by \n Priority \n Class ', anchor=tk.CENTER,
                 font=tkinter.font.Font(family='Helvetica', size=10)).grid(row=1, column=0)

        ttk.Label(self.advanced, text = 'Interarrival Times', font = tkinter.font.Font(family='Helvetica', size=10)).grid(row=0, column=1)
        self.a_list = tk.Listbox(self.advanced, width = 25, background = '#bcbcbc')
        self.a_list.grid(row=1, column=1)
        self.a_delete = ttk.Button(self.advanced, text = 'delete')
        self.a_delete.grid(row=2, column=1)
        ttk.Label(self.advanced, text='    ').grid(row=1, column=2)  # padding

        ttk.Label(self.advanced, text='Service Workloads', font=tkinter.font.Font(family='Helvetica', size=10)).grid(row=0, column=3)
        self.s_list = tk.Listbox(self.advanced, width = 25, background = '#bcbcbc')
        self.s_list.grid(row=1, column=3)
        self.s_delete = ttk.Button(self.advanced, text = 'delete')
        self.s_delete.grid(row=2, column=3)
        ttk.Label(self.advanced, text='    ').grid(row=1, column=4)  # padding

        ttk.Label(self.advanced, text='Servers', font=tkinter.font.Font(family='Helvetica', size=10)).grid(row=0, column=5)
        self.serverslist = tk.Listbox(self.advanced, width=20, background = '#bcbcbc')
        self.serverslist.grid(row=1, column=5)
        self.server_delete = ttk.Button(self.advanced, text = 'delete')
        self.server_delete.grid(row=2, column=5)

        #generate jobs and servers
        ttk.Label(self, text='').grid(row=12, column=0)  # padding
        self.generate = ttk.Button(self, text = 'Generate Jobs and Servers',  width = 25)
        self.generate.grid(row=13, column=0)
        ttk.Label(self, text='').grid(row=14, column=0)  # padding

        self.plotting = ttk.Frame(self)
        self.plotting.grid(row=15, column=0, columnspan=10)
        ttk.Label(self.plotting, text='Plot' + ' '*8, font=tkinter.font.Font(family='Helvetica', size=10, weight='bold')).grid(row=0, column=0)  # padding
        self.plot_a = ttk.Button(self.plotting, text='Arrival Times', width = 20)
        self.plot_a.grid(row=0, column=1)
        ttk.Label(self.plotting, text='    ').grid(row=0, column=2)  # padding
        self.plot_w = ttk.Button(self.plotting, text='Service Workloads', width = 20)
        self.plot_w.grid(row=0, column=3)
        ttk.Label(self.plotting, text='    ').grid(row=0, column=5)  # padding
        self.plot_k = ttk.Button(self.plotting, text='Priority Class Count', width = 20)
        self.plot_k.grid(row=0, column=6)

        #simulation
        ttk.Label(self, text='\n').grid(row=16, column=0)  # padding
        self.generate = ttk.Button(self, text = 'Run Simulation', width = 25)
        self.generate.grid(row=17, column=0)

    def selectMG_1(self, event): #disables distribution selections
        MG = event.widget.get()
        if MG == 'M (Markovian)':
            self.dis1.current(0) #lock in M
            self.dis1.configure(state = 'disabled')
        else: #G
            self.dis1.configure(state = 'readonly')

    def selectMG_2(self, event): #disables distribution selections
        MG = event.widget.get()
        if MG == 'M (Markovian)':
            self.dis2.current(0) #lock in M
            self.dis2.configure(state = 'disabled')
        else:
            self.dis2.configure(state = 'readonly')

if __name__ == '__main__':
    root = ThemedTk(theme="plastik")
    root.title(string='Queuing Simulations')
    root.geometry('800x700')
    app = QS(master=root)
    app.mainloop()
