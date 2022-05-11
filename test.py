#for testing only

from util import *
from distribution import *

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
