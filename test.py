#for testing only

from util import *
from distribution import *
import heapq

#%% 1 Job

J = Job(1, 2, 0, 'test')

print(J)

J1 = Job(2, 2)
J2 = Job(1, 2)
print(J1 < J2)

J1 = Job(2, 2)
J2 = Job(2, 2)
print(J1 < J2)

J1 = Job(2, 2)
J2 = Job(2, 2)
print(J1 <= J2)

J1 = Job(2, 2)
J2 = Job(2, 2)
print(J2 >= J1)

#higher priority comes first
J1 = Job(2, 2, 2)
J2 = Job(2, 2, 0)
print(J2 < J1)
print(J2 <= J1)


#%% 2 Server

S = Server()
print(S)

S.currentJob = J
S.starttime = 2
S.endtime = 5
print(S)

S.update_status(10, J)
print(S)

S.update_status(15)
print(S)


#%% 3 Distribution

#1
distribution = dis()
print(distribution)

np.random.seed(0)
distribution.generate_function()

a, b = distribution.generate_samples(5)
sum(a) == b[-1]
distribution.generate_samples(5, seed = 10)

distribution.generate_samples(time_end = 10, seed = None)

print(distribution.mean, np.mean(distribution.generate_samples(1000)[0]))
print(distribution.var, np.var(distribution.generate_samples(1000)[0]))

distribution = dis("exp", (2,))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

#2
distribution = dis("gamma", (1,2))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("gamma", (2,2))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("gamma", (2,5))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

#3
distribution = dis("beta", (2,2))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("beta", (1,2))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("beta", (2,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("beta", (1,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("beta", (1,1), scale = 5)
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

#4
distribution = dis("chisq", (1,))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("chisq", (10,))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

#5
distribution = dis("uniform", (0,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("uniform", (1,10))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

#6
distribution = dis("norm", (0,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("norm", (1,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

#7
distribution = dis("lognorm", (0,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("lognorm", (2,0.1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

#8
distribution = dis("weibull", (1,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("weibull", (2,2))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

#9
distribution = dis("rayleigh", 3)
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("rayleigh", 5)
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("rayleigh", 2, scale = 10)
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))


#%% 4 JobList

#random number generation test
seed = 0
scale = (1,1)
interarrivals = [("exponential", 2), ("norm", (2,1)), ("rayleigh", 1)]
workloads = [("exponential", 5), ("weibull", (1,2)), ("uniform", (0,0.5))]

time_end = 2000

arrivals, service_workloads = [], []  # will be filled for each class
for k in range(len(interarrivals)):  # priority classes
    # arrivals
    dis_name, parameters = interarrivals[k]
    arr_dis = dis(dis_name = dis_name, parameters = parameters, scale = scale[0])
    _, arrivals_element = arr_dis.generate_samples(n = n, time_end = time_end, seed = seed)
    arrivals.append(arrivals_element)
    # workloads
    dis_name, parameters = workloads[k]
    w_dis = dis(dis_name = dis_name, parameters = parameters, scale = scale[1])
    workload_element, _ = w_dis.generate_samples(n = len(arrivals_element), seed = seed, cumsum=False)  # by number of arrivals
    service_workloads.append(workload_element)

print(len(service_workloads), len(arrivals))
print(len(service_workloads[0]), len(arrivals[0]))
print(len(service_workloads[1]), len(arrivals[1]))
print(len(service_workloads[2]), len(arrivals[2]))

distribution = dis("exp", (5,))
print(distribution.mean, np.mean(service_workloads[0]))

distribution = dis("weibull", (1,2))
print(distribution.mean, np.mean(service_workloads[1]))

distribution = dis("rayleigh", (1,))
a = [arrivals[2][0]]
for i in range(1, len(arrivals[2])): #arrivals back to interarrivals
    a.append(arrivals[2][i] - arrivals[2][i-1])
print(distribution.mean, np.mean(a))

#sorting

h = []
for k in range(len(arrivals)): #a, w, k into Jobs
    for a, w in zip(arrivals[k], service_workloads[k]):
        heapq.heappush(h, Job(a, w, k))
jobs_sorted = [heapq.heappop(h) for i in range(len(h))]

akw = [(j.a, j.k, j.w) for j in jobs_sorted]

#check for sorting
akw == sorted(akw, key = lambda x: (x[0], x[1]))

#ties in arrival time

arrivals = [[1,2,5],[2,3],[2,3]]
service_workloads = [[1,2,3],[3,2],[1,1]]

h = []
for k in range(len(arrivals)): #a, w, k into Jobs
    for a, w in zip(arrivals[k], service_workloads[k]):
        heapq.heappush(h, Job(a, w, k))
jobs_sorted = [heapq.heappop(h) for i in range(len(h))]

akw = [(j.a, j.k, j.w) for j in jobs_sorted]

print(akw)



