#Utility

import numpy as np
import heapq
import random
from itertools import cycle
from collections import Counter

import pickle
import os

from math import factorial, sqrt

import warnings

import matplotlib.pyplot as plt
plt.style.use("ggplot")

import sys
import logging
logger = logging.getLogger()  # root logger
logger.setLevel(logging.INFO)

from distribution import *

#%%
class Job(object):
    '''a job object with arrival time, service workload and priority class'''
    def __init__(self, a, w, k = 0, name = "", decimals = 2): #arrival time, service workload and priority class
        if a < 0:
            raise Exception("arrival time may not be negative")
        if w <= 0:
            raise Exception("service workload must be positive")
        if type(k) != int or k < 0:
            raise Exception("priority class must a non-negative integer")

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
            raise Exception("service rate must be positive")

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
        n: number of jobs generated for each priority class; relevant only in random mode

        time_end: end time to generate jobs with arrival up to; relevant only in random mode

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

        seed: random seed
        '''

        if seed != None: #setting seed
            random.seed(seed)
            np.random.seed(seed)

        if scale[0] <= 0 or scale[1] <= 0:
            raise Exception("scaling factors must be nonnegative")
        if mode not in ["trace", "random"]:
            raise Exception("mode must be either trace or random")
        if len(interarrivals) != len(workloads):
            raise Exception("interarrivals and workloads must be of the same length")

        self.mode = mode

        if mode == "random": #need to generate random interarrival times and service workloads

            if type(interarrivals) == tuple: #one priority class only
                interarrivals = [interarrivals]
            if type(workloads) == tuple: #one priority class only
                workloads = [workloads]
            self.n_class = len(interarrivals) #number of priority classes

            arrivals, job_interarrivals, service_workloads = [], [], [] #will be filled for each class
            interarrivals_dis, service_dis = [], [] #will be filled of each class
            for k in range(len(interarrivals)): #priority classes
                #arrivals
                dis_name, parameters = interarrivals[k]
                arr_dis = dis(dis_name = dis_name, parameters = parameters, scale = scale[0])
                interarrivals_dis.append((arr_dis.mean, arr_dis.var))
                interarrivals_element, arrivals_element = arr_dis.generate_samples(n = n, time_end = time_end, seed = None)
                job_interarrivals.append(interarrivals_element) #the sole purpose of this is for trace saving
                arrivals.append(arrivals_element) #arrival times
                #workloads
                dis_name, parameters = workloads[k]
                w_dis = dis(dis_name = dis_name, parameters = parameters, scale = scale[1])
                service_dis.append((w_dis.mean, w_dis.var))
                workload_element, _ = w_dis.generate_samples(n = len(arrivals_element), seed = None, cumsum = False) #by number of arrivals
                service_workloads.append(workload_element) #service workloads
            self.interarrivals_dis = interarrivals_dis
            self.service_dis = service_dis

        else: #trace mode
            job_interarrivals = interarrivals.copy()
            self.n_class = len(interarrivals)  #number of priority classes
            arrivals = [] #only preparing the arrivals from interarrivals
            for interarrivals_element in interarrivals: #interarrivals_element contains interrrivals for a class
                arrivals_element = [interarrivals_element[0]]
                for e in interarrivals_element[1:]: #cumsum
                    arrivals_element.append(arrivals_element[-1] + e)
                arrivals.append(arrivals_element)
                service_workloads = workloads #just to prepare for getting sorted job list
            self.interarrivals_dis, self.service_dis = None, None #not applicable

        self.interarrival_trace = job_interarrivals
        self.workload_trace = service_workloads

        h = [] #initialise for heap
        for k in range(len(arrivals)):  #a, w, k into Jobs
            for a, w in zip(arrivals[k], service_workloads[k]):
                heapq.heappush(h, Job(a, w, k))
        self.jobs = [heapq.heappop(h) for i in range(len(h))] #sorted by arrival and then priority class
        for i, j in enumerate(self.jobs):
            j.name = i #tagging all jobs to keep track of them in terms of order of arrival and priority class
        self.a, self.w, self.k = [], [], []
        for j in self.jobs:
            self.a.append(j.a)
            self.w.append(j.w)
            self.k.append(j.k)

    @staticmethod
    def create_from_file(dir = '.', folder_name = 'JobList', scale = (1,1)): #loading in interarrivals and workloads and create
        path = dir + '/' + folder_name
        with open(path + '/interarrivals', "rb") as f:
            interarrival_trace = pickle.load(f)
        with open(path + '/workload', "rb") as f:
            workload_trace = pickle.load(f)
        return JobList(mode = "trace", interarrivals = interarrival_trace, workloads = workload_trace,
                       scale = scale)

    def save_trace(self, dir = '.', folder_name = 'JobList'): #save interarrivals and workloads
        path = dir + '/' + folder_name
        if not os.path.exists(path):
            os.mkdir(path)
        with open(path + '/interarrivals', "wb") as f:
            pickle.dump(self.interarrival_trace, f)
        with open(path + '/workload', "wb") as f:
            pickle.dump(self.workload_trace, f)

    def plot_a(self): #time series plot of arrival times
        fig = plt.figure()
        k_adj = [] #with -0.1, 0, 0.1 displacements
        cycles = [cycle([-0.1, 0, 0.1]) for _ in range(self.n_class)]
        for i in range(len(self.k)): #displacing points vertically slightly to each class
            k = self.k[i]
            k_adj.append(k + next(cycles[k]))
        plt.scatter(self.a, k_adj, c = self.k, cmap = "cool", alpha = 0.75); #colouring by class
        plt.yticks(range(self.n_class)); #only the priority classes
        plt.ylim(-0.5, self.n_class - 0.5); #y axis limits
        plt.ylabel("priority class");
        plt.xlabel("time");
        plt.title("Arrival Times");
        plt.show()
        return fig

    def plot_w(self): #histogram plot of work loads
        fig = plt.figure()
        plt.hist([[w for w, c in zip(self.w, self.k) if c == k] for k in range(self.n_class)], #workloads by class
                 alpha = 0.75, label = range(self.n_class));
        plt.legend(loc = "upper right", title = "Priority Class");
        plt.xlabel("service workload");
        plt.title("Service Workload Distribution");
        plt.show()
        return fig

    def plot_k(self): #count by priority class
        fig = plt.figure()
        class_count = Counter(self.k)
        plt.bar(range(self.n_class), [class_count[k] for k in range(self.n_class)], align = "center", alpha = 0.75);
        plt.xlabel("priority class");
        plt.xticks(range(self.n_class));
        plt.title("Job Count by Priority Class");
        plt.show()
        return fig

#%%
class Simulation(object):
    '''a queueing simulation'''
    def __init__(self, JobList, Servers = [Server()], maxtime = np.Inf): #simulate starts with all servers idle
        self.statistics = {} #to be filled upon simulation run
        self.JobList = JobList
        self.Servers = Servers #when there are multiple idle servers, the one with min index is prioritised for processing
        self.n_servers = len(Servers)
        if maxtime <= 0:
            raise Exception("max time must be positive")
        self.maxtime = maxtime

    def run(self, logfile = "", printlog = False, comprehensive_print = False, decimals = 5, server_assign = "random"): #run simulations
        '''run simulations'''
        ##### I. Setup
        #setting up the loggers
        logger.handlers = [] #cleaning up handlers to avoid duplicated printing

        JL = self.JobList
        queues = [[] for k in range(JL.n_class)] #separate queue for each class
        jobs = JL.jobs + [Job(np.Inf, np.Inf, 0, "never")]
        masterclock = 0 #initialise the time, simulations stop upon reaching max time or completing all jobs
        Servers = self.Servers.copy()

        #set up for response, waiting and service time by priority class and server busy times
        #also count for jobs completed, started and still waiting
        self.statistics = {"server_busy_time" : [[] for _ in range(len(Servers))],
                           "waiting_times" : [0 for j in JL.jobs],
                           "service_times": [0 for j in JL.jobs], #keeping track of this when they depart as servers might have different efficiencies
                           "jobs_completed" : [0 for _ in range(JL.n_class)], "jobs_in_server" : [0 for _ in range(JL.n_class)], "jobs_in_queue" : [0 for _ in range(JL.n_class)]
                           }

        if printlog: #print in console
            logger.addHandler(logging.StreamHandler(sys.stdout))

        if logfile: #print in the log if provided
            logfile += '' if logfile.endswith(".log") else ".log"
            logger.addHandler(logging.FileHandler(logfile))

        log_flag = bool(logfile or printlog) #whether to print log at all
        if log_flag:
            logger.info("#" * 50 + "  simulation starts  ")
            if self.maxtime < np.Inf:
                logger.info(f"time up to {self.maxtime :.{decimals}f}\n")
            else:
                logger.info(f"until all jobs completed\n")
            logger.info("=" * 10 + f" masterclock {masterclock :.{decimals}f}")
            logger.info("")

        ##### II. Simulation Loop
        while masterclock <= self.maxtime:
            #advance in time
            potential_events = [(jobs[0].a, self.n_servers)] + [(Servers[i].endtime, i) for i in range(self.n_servers)] #next arrival or job completion at a processor
            #next event
            event_time, event_type = min(potential_events, key = lambda x: (x[0], x[1]))  #departure is prioritised

            if event_time == np.inf:  #no more events to come
                if log_flag:
                    logger.info("#" * 50 + "  simulation ends by completion of all arrived jobs ")
                break
            elif event_time > self.maxtime: #beyond max time for simulation
                if log_flag:
                    logger.info("#" * 50 + "  simulation ends by reaching end time {self.maxtime} ")
                break
            else: #simulation continues
                masterclock = event_time  #advance masterclock to event time
                if log_flag:
                    logger.info("=" * 10 + f" masterclock {masterclock :.{decimals}f}")

            ##### arrival event
            if event_type == self.n_servers: #arrival
                arrived_job = jobs.pop(0) #this job just arrived, it will either go into a queue or go into a server (processor)

                if log_flag:
                    logger.info(f"job arrival with workload {arrived_job.w :.{decimals}f}" +
                                (f" and priority class {arrived_job.k}" if JL.n_class > 1 else "")) #not printing priority class if there is only class

                if server_assign == "random":
                    idle_servers = [i for i, s in enumerate(Servers) if s.endtime == np.Inf]
                    if idle_servers: #at least one server idle
                        s = Servers[random.choice(idle_servers)] #random server
                        s.update_status(masterclock, arrived_job)
                        if log_flag:
                            logger.info("assigning into server" + (
                                f" {i} " if self.n_servers > 1 else " ")  # if only one server not printing the server index
                                        + f"and will finish at {s.endtime :.{decimals}f}")
                    else: #all server busy
                        if log_flag:
                            logger.info("go into queue")
                        queues[arrived_job.k].append(arrived_job) #append into the queue of the corresponding priority class
                else: #take priorities by order of servers
                    for i, s in enumerate(Servers): #iterate over all servers to see if the arrived job can be assigned
                        if s.endtime == np.Inf: #idle server found
                            s.update_status(masterclock, arrived_job)
                            if log_flag:
                                logger.info("assigning into server" + (f" {i} " if self.n_servers > 1 else " ") #if only one server not printing the server index
                                            + f"and will finish at {s.endtime :.{decimals}f}")
                            break
                    else: #did not break so go into the queue
                        if log_flag:
                            logger.info("go into queue")
                        queues[arrived_job.k].append(arrived_job) #append into the queue of the corresponding priority class

            ##### departure event
            else: #departure (event_type is the index of the server)
                if log_flag:
                    logger.info("departure at server" + (f" {event_type}" if self.n_servers > 1 else "") #if only one server not printing the server index
                                )
                s = Servers[event_type] #server of interest
                self.statistics["server_busy_time"][event_type].append((s.starttime, s.endtime))  #update busy time period
                job_index = s.currentJob.name #this is the index in the job list
                self.statistics["waiting_times"][job_index] = (s.starttime - s.currentJob.a)
                self.statistics["service_times"][job_index] = (s.endtime - s.starttime) #equivalent to server busy time
                self.statistics["jobs_completed"][s.currentJob.k] += 1

                for i, q in enumerate(queues): #iterate over queues of all priority classes in priority order
                    if q: #this queue is not empty
                        if log_flag:
                            logger.info("assigning first in the queue " +
                                        (f"of priority class {arrived_job.k} to the server" if JL.n_class > 1 else "to the server")) #not printing priority class if there is only class
                        assign_job = q.pop(0) #first one in the queue
                        s.update_status(masterclock, assign_job)
                        break
                else: #did not break so no one in the queue
                    if log_flag:
                        logger.info("the server becomes idle")
                    s.update_status(masterclock) #the server becomes idle

            if comprehensive_print and log_flag: #comprehensive print
                for i, s in enumerate(Servers): #servers
                    s_status = f"finishing at {s.endtime :.{decimals}f}" if s.endtime < np.Inf else "idle"
                    logger.info("server" + (f" {i}" if self.n_servers > 1 else "") + f": {s_status}")
                for i, q in enumerate(queues): #queues
                    q_status = [(float(f"{j.a :.{decimals}f}"), float(f"{j.w :.{decimals}f}")) for j in q]
                    logger.info("queue" + (f" {i}" if JL.n_class > 1 else "") + f": {q_status}")

        ##### III. Additional Summary
        self.statistics["response_times"] = (np.array(self.statistics["waiting_times"]) + np.array(self.statistics["service_times"])).tolist()
        self.statistics["final_masterclock"] = masterclock #final time before exceeding max time or run out of upcoming events

        #remaining ones
        for s in Servers:
            if s.currentJob: #busy servers
                self.statistics["jobs_in_server"][s.currentJob.k] += 1
        for i in range(len(queues)):
            self.statistics["jobs_in_queue"][i] = len(queues[i])

    def evaluate(self, exclusion = 0.25): #evaluate results from simulations
        '''evaluate core performance of the system'''
        if exclusion < 0 or exclusion >= 1:
            raise Exception('initial transient data exclusion must be a value between 0 and 1')
        if not self.statistics: #simulation yet to be run
            self.run()
        cutoff_time = self.statistics["final_masterclock"] * exclusion
        self.cutoff_time = cutoff_time
        time_period = self.statistics["final_masterclock"] - cutoff_time
        JL = self.JobList
        #average response times, waiting times and service times by class
        (self.statistics['avg_response_times'],
         self.statistics['avg_waiting_times'],
         self.statistics['avg_service_times']
         ) = {i:[] for i in range(JL.n_class)}, {i:[] for i in range(JL.n_class)}, {i:[] for i in range(JL.n_class)}
        for rt, wt, st, k in zip(self.statistics["response_times"], self.statistics["waiting_times"], self.statistics["service_times"],
                                 JL.k): #reorganise by priority class
            self.statistics['avg_response_times'][k].append(rt)
            self.statistics['avg_waiting_times'][k].append(wt)
            self.statistics['avg_service_times'][k].append(st)
        class_count = Counter(JL.k)
        for k in range(JL.n_class):
            start = round(class_count[k] * exclusion)
            self.statistics['avg_response_times'][k] = np.average(self.statistics['avg_response_times'][k][start:])
            self.statistics['avg_waiting_times'][k] = np.average(self.statistics['avg_waiting_times'][k][start:])
            self.statistics['avg_service_times'][k] = np.average(self.statistics['avg_service_times'][k][start:])
        #utilisation of servers
        self.statistics['server_utilisation'] = [0 for _ in range(len(self.statistics['server_busy_time']))]
        for s in range(len(self.statistics['server_busy_time'])):
            busy_periods = self.statistics['server_busy_time'][s]
            for start_time, end_time in busy_periods:
                if end_time <= cutoff_time:
                    continue
                elif (end_time > cutoff_time) and (start_time <= cutoff_time): #started before cutoff time but ended after it
                    self.statistics['server_utilisation'][s] += ((end_time - cutoff_time) / time_period)
                else:
                    self.statistics['server_utilisation'][s] += ((end_time - start_time) / time_period)

def ErlangC(c, lamb, mu):
    '''Erlang C Formula'''
    rho = lamb/(c * mu)
    denominator = 1 + (1 - rho) * (factorial(c) / (c * rho)**c) * np.sum([ (c * rho)**k / factorial(k) for k in range(c)])
    return 1 / denominator

#%%
class MM1(Simulation): #single class
    '''an MM1 queueing simulation'''
    def __init__(self, lamb, mu, n = 100, time_end = None, seed = 0, scale = (1, 1), maxtime = np.Inf):
        self.lamb, self.mu = lamb, mu
        if lamb >= mu:
            warnings.warn("instable queue", Warning)
        else:
            self.rho = lamb/mu
            self.expected_response_time = 1 / (mu - lamb)
        JL = JobList(n = n, time_end = time_end, mode = "random", interarrivals = ("exponential", lamb), workloads = ("exponential", mu),
                     scale = scale, seed = seed)
        Simulation.__init__(self, JobList = JL, Servers = [Server()], maxtime = maxtime)

#%%
class MMn(Simulation): #single class all servers have the same efficiency (1) so service workload is service time
    '''an MMn queueing simulation'''
    def __init__(self, lamb, mu, n_servers, n = 100, time_end = None, seed = 0, scale = (1, 1), maxtime = np.Inf):
        self.lamb, self.mu, self.n_servers = lamb, mu, n_servers
        if lamb >= (n_servers * mu):
            warnings.warn("instable queue", Warning)
        else:
            self.rho = lamb/(n_servers * mu)
            self.expected_response_time = ErlangC(n_servers, lamb, mu) / (n_servers * mu - lamb) + 1/mu
        JL = JobList(n = n, time_end = time_end, mode = "random", interarrivals = ("exponential", lamb), workloads = ("exponential", mu),
                     scale = scale, seed = seed)
        Simulation.__init__(self, JobList = JL, Servers = [Server() for _ in range(n_servers)], maxtime = maxtime)

#%%
class GGn(Simulation): #single class all servers have the same efficiency (1) so service workload is service time
    '''an GGn queueing simulation'''
    def __init__(self, interarrival_time, service_workload, n_servers, n = 100, time_end = None, seed = 0, scale = (1, 1), maxtime = np.Inf):
        dis_name, dis_parameters = interarrival_time
        interarrival_distribution = dis(dis_name, dis_parameters)
        dis_name, dis_parameters = service_workload
        service_distribution = dis(dis_name, dis_parameters)
        lamb = 1 / interarrival_distribution.mean #arrival rate
        mu = 1 / service_distribution.mean #service rate
        CV_a, CV_s = sqrt(interarrival_distribution.var) / interarrival_distribution.mean, sqrt(service_distribution.var) / service_distribution.mean #coefficient of variation
        self.expected_response_time = ErlangC(n_servers, lamb, mu) / (n_servers * mu - lamb) * (CV_a**2 + CV_s**2) / 2 + service_distribution.mean #only approximation
        JL = JobList(n = n, time_end = time_end, mode = "random", interarrivals = interarrival_time, workloads = service_workload,
                     scale = scale, seed = seed)
        Simulation.__init__(self, JobList = JL, Servers = [Server() for _ in range(n_servers)], maxtime = maxtime)