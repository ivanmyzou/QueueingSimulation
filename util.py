#Utility

import numpy as np

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
    def __init__(self, mode = "random", interarrivals = ("exponential", 1), workloads = ("exponential", 1), scale = (1, 1)): #trace mode or random mode
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
        pass

#%%
class Simulation(object):
    '''a queueing simulation'''
    def __init__(self, JobList, Servers, maxtime = np.Inf): #simulate starts with all servers idle
        pass

    def run(self): #run simulations
        pass

    def evaluate(self): #evaluate results from simulations
        pass
