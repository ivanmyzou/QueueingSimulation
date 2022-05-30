#for testing only

from util import *
from distribution import *
import heapq

#====================================================================================================
#====================================================================================================
#%% I. Job

# 1.1 basics
J = Job(1, 2, 0, 'test')
print(J)

# 1.2 comparisons
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

# 1.3 more comparisons
#higher priority comes first
J1 = Job(2, 2, 2)
J2 = Job(2, 2, 0)
print(J2 < J1)
print(J2 <= J1)

# 1.4 special case
#this to represent no more job arrivals
Jnever = Job(np.Inf, np.Inf, 0, 'never')
print(Jnever)

#====================================================================================================
#====================================================================================================
#%% II Server

# 2.1 basics
S = Server()
print(S)
print(S.endtime, S.starttime)

S.currentJob = J
S.starttime = 2
S.endtime = 5
print(S)

# 2.2 updates
S.update_status(10, J)
print(S)
print(S.endtime, S.starttime)

S.update_status(15)
print(S)
print(S.endtime, S.starttime)

#====================================================================================================
#====================================================================================================
#%% III Distribution

# 3.1.1 exp
distribution = dis()
print(distribution)

np.random.seed(0)
distribution.generate_function()

a, b = distribution.generate_samples(5)
sum(a) == b[-1]
distribution.generate_samples(5, seed = 10)

distribution.generate_samples(time_end = 10, seed = None)

# 3.1.2 mean & var
print(distribution.mean, np.mean(distribution.generate_samples(1000)[0]))
print(distribution.var, np.var(distribution.generate_samples(1000)[0]))

distribution = dis("exp", (2,))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

# 3.2 gamma
distribution = dis("gamma", (1,2))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("gamma", (2,2))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("gamma", (2,5))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

# 3.3 beta
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

# 3.4 chisq
distribution = dis("chisq", (1,))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("chisq", (10,))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

# 3.5 unif
distribution = dis("uniform", (0,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("uniform", (1,10))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

# 3.6 norm
distribution = dis("norm", (0,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("norm", (1,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

# 3.7 lognorm
distribution = dis("lognorm", (0,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("lognorm", (2,0.1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

# 3.8 weibull
distribution = dis("weibull", (1,1))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("weibull", (2,2))
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

# 3.9 rayleigh
distribution = dis("rayleigh", 3)
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("rayleigh", 5)
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

distribution = dis("rayleigh", 2, scale = 10)
print(distribution.mean, np.mean(distribution.generate_samples(5000)[0]))
print(distribution.var, np.var(distribution.generate_samples(5000)[0]))

# 3.10 reproducing
distribution = dis("exp", (2,))
sum(distribution.generate_samples(10)[0])

np.random.seed(0)
sum(distribution.generate_samples(10, seed = None)[0])

#====================================================================================================
#====================================================================================================
#%% IV JobList

# 4.1 internal logic
#random number generation test
seed = 0
scale = (1,1)
interarrivals = [("exponential", 2), ("norm", (2,1)), ("rayleigh", 1)]
workloads = [("exponential", 5), ("weibull", (1,2)), ("uniform", (0,0.5))]

n = None
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

# 4.2.1 sorting with heaps
#sorting
h = []
for k in range(len(arrivals)): #a, w, k into Jobs
    for a, w in zip(arrivals[k], service_workloads[k]):
        heapq.heappush(h, Job(a, w, k))
jobs_sorted = [heapq.heappop(h) for i in range(len(h))]
akw = [(j.a, j.k, j.w) for j in jobs_sorted]

#check for sorting
print(akw == sorted(akw, key = lambda x: (x[0], x[1])))

# 4.2.2 more sorting cases
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

# 4.3 plotting
JL = JobList()
fig = JL.plot_a()
fig = JL.plot_w()
fig = JL.plot_k()

JL = JobList(time_end = 200, interarrivals = [('exp', 0.5), ('exp', 2)], workloads = [('normal', (1, 1)), ('exp', 1)])
fig = JL.plot_a()
fig = JL.plot_w()
fig = JL.plot_k()

# 4.4.1 trace save
JL = JobList(25)
print(sum(JL.a))
print(sum(JL.w))
JL.save_trace(dir = '.', folder_name = 'test')

# 4.4.2 trace import
JL_imported = JobList.create_from_file(dir = '.', folder_name = 'test')
print(sum(JL_imported.a))
print(sum(JL_imported.w))

# 4.4.3 trace reconciliation
#checking all jobs
print(all([(a.a == b.a) and (a.w == b.w) and (a.k == b.k)
           for a,b in zip(JL.jobs, JL_imported.jobs)]))

#====================================================================================================
#====================================================================================================
#%% V Simulation

# 5.1 basics
sim = Simulation(JL)
sim.run(printlog = True)

# 5.2.1 more complicated scenarios
JL = JobList(25, interarrivals = [('exp', 0.5), ('exp', 2)], workloads = [('normal', (1, 1)), ('exp', 1)])
print(sum(JL.a))
print(sum(JL.w))
JL.save_trace(dir = '.', folder_name = 'test')

JL_imported = JobList.create_from_file(dir = '.', folder_name = 'test')
print(sum(JL_imported.a))
print(sum(JL_imported.w))

#checking all jobs
print(all([(a.a == b.a) and (a.w == b.w) and (a.k == b.k)
           for a,b in zip(JL.jobs, JL_imported.jobs)]))

# 5.2.2 more tests
#test simulation
JL = JobList(time_end = 25, interarrivals = [('exp', 0.5), ('exp', 2)], workloads = [('normal', (1, 1)), ('exp', 1)])
sim = Simulation(JL, [Server() for _ in range(2)])

sim.run(logfile = 'test.log', printlog = True, comprehensive_print = True)

# 5.2.3 more tests
#test simulation
JL = JobList(time_end = 25, interarrivals = [('exp', 0.5), ('exp', 2)], workloads = [('normal', (1, 1)), ('exp', 1)])
sim = Simulation(JL, [Server() for _ in range(3)])

sim.run(logfile = 'test.log', printlog = True, comprehensive_print = True)

# 5.2.4 more tests
#test simulation
JL = JobList(mode = 'trace',
             interarrivals = [[1, 4, 2, 2], [3, 0, 3, 0, 2, 0]],
             workloads = [[1.8, 3.9, 1.9, 3.8], [8, 6.1, 2.1, 3.1, 5, 4.1]])
print(JL.a)
sim = Simulation(JL, [Server() for _ in range(4)])
sim.run(printlog = True, comprehensive_print = True, decimals = 2)

# 5.2.5 more tests
#test simulation
JL = JobList(time_end = 2000, interarrivals = ('exp', 1), workloads = ('exp', 3))
sim = Simulation(JL, [Server()])

sim.run(printlog = True, comprehensive_print = True)
print(np.mean(sim.statistics['response_times'][500:]))
print(len(sim.statistics['server_busy_time'][0]))
print(sim.statistics['jobs_completed'])
print(sim.statistics['jobs_in_server'])
print(sim.statistics['jobs_in_queue'])

mm1_sim = MM1(1, 2, time_end = 2000, seed = 1)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]))

mm1_sim = MM1(1, 2, time_end = 2000, seed = 2)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]))

mm1_sim = MM1(1, 2, time_end = 2000, seed = 3)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]))

# 5.3 empirical vs theoretical
mm1_sim = MM1(0.75, 2, time_end = 2000, seed = 0)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]), mm1_sim.expected_response_time)
mm1_sim.evaluate()
print(mm1_sim.statistics['server_utilisation'], mm1_sim.rho)

mm1_sim = MM1(0.75, 2, time_end = 2000, seed = 1)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]), mm1_sim.expected_response_time)
mm1_sim.evaluate()
print(mm1_sim.statistics['server_utilisation'], mm1_sim.rho)

mm1_sim = MM1(0.75, 2, time_end = 2000, seed = 2)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]), mm1_sim.expected_response_time)
mm1_sim.evaluate()
print(mm1_sim.statistics['server_utilisation'], mm1_sim.rho)

mm1_sim = MM1(0.75, 2, time_end = 5000, seed = 0)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]), mm1_sim.expected_response_time)
mm1_sim.evaluate()
print(mm1_sim.statistics['server_utilisation'], mm1_sim.rho)

mm1_sim = MM1(1, 1.5, time_end = 2000, seed = 0)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]), mm1_sim.expected_response_time)
mm1_sim.evaluate()
print(mm1_sim.statistics['server_utilisation'], mm1_sim.rho)

mm1_sim = MM1(1, 1.5, time_end = 2000, seed = 1)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][500:]), mm1_sim.expected_response_time)
mm1_sim.evaluate()
print(mm1_sim.statistics['server_utilisation'], mm1_sim.rho)

mm1_sim = MM1(1, 1.5, time_end = 2000, seed = 2)
mm1_sim.run()
print(np.mean(mm1_sim.statistics['response_times'][1000:]), mm1_sim.expected_response_time)
mm1_sim.evaluate()
print(mm1_sim.statistics['server_utilisation'], mm1_sim.rho)

mmn_sim = MMn(1, 1, 2, time_end = 2000, seed = 0)
mmn_sim.run()
print(np.mean(mmn_sim.statistics['response_times'][1000:]), mmn_sim.expected_response_time)

mmn_sim = MMn(1, 1, 2, time_end = 2000, seed = 1)
mmn_sim.run()
print(np.mean(mmn_sim.statistics['response_times'][1000:]), mmn_sim.expected_response_time)

mmn_sim = MMn(1, 0.5, 5, time_end = 2000, seed = 0)
mmn_sim.run()
print(np.mean(mmn_sim.statistics['response_times'][1000:]), mmn_sim.expected_response_time)

mmn_sim = MMn(1, 0.5, 4, time_end = 2000, seed = 0)
mmn_sim.run()
print(np.mean(mmn_sim.statistics['response_times'][1000:]), mmn_sim.expected_response_time)

mmn_sim = MMn(1, 0.5, 4, time_end = 2000, seed = 1)
mmn_sim.run()
print(np.mean(mmn_sim.statistics['response_times'][1000:]), mmn_sim.expected_response_time)
mmn_sim.evaluate()
print(mmn_sim.statistics['server_utilisation'], mmn_sim.rho)

mmn_sim = MMn(1, 0.5, 4, time_end = 2000, seed = 2)
mmn_sim.run()
print(np.mean(mmn_sim.statistics['response_times'][1000:]), mmn_sim.expected_response_time)
mmn_sim.evaluate()
print(mmn_sim.statistics['server_utilisation'], mmn_sim.rho)

mmn_sim = MMn(1, 0.5, 5, time_end = 4000, seed = 2)
mmn_sim.run()
print(np.mean(mmn_sim.statistics['response_times'][1000:]), mmn_sim.expected_response_time)
mmn_sim.evaluate()
print(mmn_sim.statistics['server_utilisation'], mmn_sim.rho)

# 5.4 GGn

ggn = GGn(("uniform", (0,2)), ("uniform", (0,1)), 3, time_end = 2000, seed = 0)
ggn.run()
print(np.mean(ggn.statistics['response_times'][1000:]), ggn.expected_response_time)

ggn = GGn(("uniform", (0,1)), ("uniform", (0,1)), 2, time_end = 5000, seed = 0)
ggn.run()
print(np.mean(ggn.statistics['response_times'][1000:]), ggn.expected_response_time)

ggn = GGn(("uniform", (0,1)), ("beta", (1,1)), 5, time_end = 5000, seed = 0)
ggn.run()
print(np.mean(ggn.statistics['response_times'][1000:]), ggn.expected_response_time)

ggn = GGn(("normal", (0,1)), ("beta", (1,2)), 8, time_end = 5000, seed = 0)
ggn.run()
print(np.mean(ggn.statistics['response_times'][1000:]), ggn.expected_response_time)

#====================================================================================================
#====================================================================================================
#%% VI Evaluations

# 6.1 basics
JL = JobList(time_end = 500, interarrivals = [('exp', 0.5), ('exp', 2)], workloads = [('normal', (1, 1)), ('exp', 1)])
sim = Simulation(JL, [Server() for _ in range(4)])

sim.evaluate()

print(sim.statistics.keys())

print(sim.statistics['avg_response_times'])
print(sim.statistics['avg_waiting_times'])
print(sim.statistics['avg_service_times'])

# 6.2 plots
JL = JobList(time_end = 1000,
             interarrivals = [('exp', 1), ('exp', 0.5), ('exp', 0.5)],
             workloads = [('exp', 0.5), ('exp', 0.5), ('exp', 0.5)])
sim = Simulation(JL, [Server() for _ in range(5)])
sim.evaluate()

len(sim.statistics['response_times']), len(sim.JobList.k)
len(sim.statistics['waiting_times']), len(sim.JobList.k)

sim.plot_r()
sim.plot_w()

