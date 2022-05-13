#Utility

import numpy as np
import heapq
from itertools import cycle
from collections import Counter

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from distribution import *

#%%
class Job(object):
    '''a job object with arrival time, service workload and priority class'''
    def __init__(self, a, w, k = 0, name = '', decimals = 2): #arrival time, service workload and priority class
        if a < 0:
            raise Exception('arrival time may not be negative')
        if w < 0:
            raise Exception('service workload may not be negative')
        if type(k) != int or k < 0:
            raise Exception('priority class must a non-negative integer')

        self.a, self.w, self.k = a, w, k
        self.name = name #job name
        self.decimals = decimals
        #service time is (w / mu) where mu is the service rate of the server (or processor)
        #this allows simulations where there are servers of different service rate

    def __str__(self):
        job_name = f"job {self.name}" if self.name else "a job"
        return f"{job_name} arriving at {self.a :.{self.decimals}f} with service workload {self.w :.{self.decimals}f} and priority class {self.k}"

    def __lt__(self, other): #order by arrival time and then priority classes
        return (self.a < other.a) or (self.a == other.a and self.k < other.k)

    def __le__(self, other): #order by arrival time and then priority classes
        return (self.a < other.a) or (self.a == other.a and self.k <= other.k)

#%%
class Server(object): #a server may process one job at a time
    '''a server with service rate'''
    def __init__(self, mu = 1, decimals = 2): #service rate defaults to 1 so that service workload corresponds to the service time
        if mu <= 0:
            raise Exception('service rate must be positive')

        self.mu = mu
        self.currentJob = None #initialise with no job being processed
        self.starttime = None #when the current job started being processed
        self.endtime = np.Inf #infinity so that it never happens
        self.decimals = decimals

    def __str__(self):
        if self.currentJob: #busy
            job_name = f"job {self.currentJob.name}" if self.currentJob.name else "a job"
            return (f"server with {job_name} started at {self.starttime :.{self.decimals}f} to be finished at {self.endtime :.{self.decimals}f}"
                    + f" with service time {self.endtime - self.starttime :.{self.decimals}f}"
                    + (f" with service rate {self.mu :.{self.decimals}f}" if self.mu != 1 else ""))
        else: #not busy
            return "a currently idle server" + (f" with service rate {self.mu :.{self.decimals}f}" if self.mu != 1 else "")

    def update_status(self, currenttime, newJob = None):
        if newJob: #arrival
            self.currentJob = newJob
            self.starttime = currenttime
            self.endtime = currenttime + (self.currentJob.w / self.mu) #current time + service time
        else: #departure
            self.currentJob = None
            self.starttime = None
            self.endtime = np.Inf

#%%
class JobList(object): #sort by (arrival time, priority class) in ascending order
    '''a list of jobs of different priority classes with information of arrival times and service workloads'''
    def __init__(self, n = 100, time_end = None, mode = "random", interarrivals = ("exponential", 1), workloads = ("exponential", 1),
                 scale = (1, 1), seed = 0): #trace mode or random mode
        '''
        mode: defaults to "random"; either "random" or "trace";
        when it is "random", generate interarrival times and service workloads based on distributions specified;
        when it is "trace", interarrival times and service workloads come from direct inputs

        interarrivals: a list with each element corresponding to a priority class
        when in random mode, for each priority class, the element is a tuple specifying the distribution and parameters
        when in trace mode, for each priority class, the element is a list with its elements representing the interarrivals

        workloads: a list with each element corresponding to a priority class
        when in random mode, for each priority class, the element is a tuple specifying the distribution and parameters
        when in trace mode, for each priority class, the element is a list with its elements representing the work loads

        scale: a tuple indicating the values to scale the generated or provided interarrivals and workloads respectively
        '''

        if scale[0] <= 0 or scale[1] <= 0:
            raise Exception('scaling factors must be nonnegative')
        if mode not in ['trace', 'random']:
            raise Exception('mode must be either trace or random')
        if len(interarrivals) != len(workloads):
            raise Exception('intearrivals and workloads must be of the same length')

        self.mode = mode
        if mode == 'random': #need to generate random interarrival times and service workloads
            if type(interarrivals) == tuple: #one priority class only
                interarrivals = [interarrivals]
            if type(workloads) == tuple: #one priority class only
                workloads = [workloads]
            self.n_class = len(interarrivals) #number of priority classes
            arrivals, service_workloads = [], [] #will be filled for each class
            for k in range(len(interarrivals)): #priority classes
                #arrivals
                dis_name, parameters = interarrivals[k]
                arr_dis = dis(dis_name = dis_name, parameters = parameters, scale = scale[0])
                _, arrivals_element = arr_dis.generate_samples(n = n, time_end = time_end, seed = seed)
                arrivals.append(arrivals_element)
                #workloads
                dis_name, parameters = workloads[k]
                w_dis = dis(dis_name = dis_name, parameters = parameters, scale = scale[1])
                workload_element, _ = w_dis.generate_samples(n = len(arrivals_element), seed = seed, cumsum = False) #by number of arrivals
                service_workloads.append(workload_element)
        else: #trace mode
            self.n_class = len(interarrivals)  # number of priority classes
            arrivals = [] #only preparing the arrivals from interarrivals
            for interarrivals_element in interarrivals: #interarrivals_element contains interrrivals for a class
                arrivals_element = [interarrivals_element[0]]
                for e in interarrivals_element[1:]: #cumsum
                    arrivals_element.append(interarrivals_element[-1] + e)
                arrivals.append(arrivals_element)
                service_workloads = workloads #just to prepare for getting sorted job list

        h = [] #initialise for heap
        for k in range(len(arrivals)):  # a, w, k into Jobs
            for a, w in zip(arrivals[k], service_workloads[k]):
                heapq.heappush(h, Job(a, w, k))
        self.jobs = [heapq.heappop(h) for i in range(len(h))] #sorted by arrival and then priority class
        self.a, self.w, self.k = [], [], []
        for j in self.jobs:
            self.a.append(j.a)
            self.w.append(j.w)
            self.k.append(j.k)

    def plot_a(self):
        fig = plt.figure()
        k_adj = [] #with -0.1, 0, 0.1 displacements
        cycles = [cycle([-0.1, 0, 0.1]) for _ in range(self.n_class)]
        for i in range(len(self.k)): #displacing points vertically slightly to each class
            k = self.k[i]
            k_adj.append(k + next(cycles[k]))
        plt.scatter(self.a, k_adj, c = self.k, cmap = "cool", alpha = 0.5); #colouring by class
        plt.yticks(range(self.n_class)); #only the priority classes
        plt.ylim(-0.5, self.n_class - 0.5); #y axis limits
        plt.ylabel('priority class');
        plt.xlabel('time');
        plt.title('Arrival Times');
        plt.show();
        return fig

    def plot_w(self):
        fig = plt.figure()
        plt.hist([[w for w, c in zip(self.w, self.k) if c == k] for k in range(self.n_class)], #workloards by class
                 alpha = 0.75, label = range(self.n_class));
        plt.legend(loc = 'upper right', title = 'Priority Class');
        plt.xlabel('service workload');
        plt.title('Service Workload Distribution');
        plt.show();
        return fig

    def plot_k(self):
        fig = plt.figure()
        class_count = Counter(self.k)
        plt.bar(range(self.n_class), [class_count[k] for k in range(self.n_class)], align = 'center', alpha = 0.75);
        plt.xlabel('priority class');
        plt.xticks(range(self.n_class));
        plt.title('Job Count by Priority Class');
        plt.show();
        return fig

#%%
class Simulation(object):
    '''a queueing simulation'''
    def __init__(self, JobList, Servers, maxtime = np.Inf): #simulate starts with all servers idle
        pass

    def run(self): #run simulations
        pass

    def evaluate(self): #evaluate results from simulations
        pass
