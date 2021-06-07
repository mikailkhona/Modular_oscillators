"""
Created on Fri Jun 26 2020 
By Francisco Acosta

Simulates 1D Kuramoto model with local coupling

saves output data as array of phase evolutions in "phase_evolution.dat"

"""
import math
import numpy as np
import scipy.sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time



""" ..........................Helper functions............................... """


## Creates NxN interaction matrix with local uniform coupling k. Aperiodic by default (set periodic = True to make periodic).
## radius determines range of local interactions. By deault radius = 1 (i.e., each oscilliator interacts only with its 2 closest neighbors)
def interaction_matrix(N,periodic = False, radius = 1):
   
    W = np.zeros((N,N))
    for i in range(-radius,radius+1):
        #vec = (i!=0)*(k/(2**(abs(i)-1)))*np.ones(N-abs(i))
        vec = (i!=0)*(1)*np.ones(N-abs(i))
        W += np.diag(vec,i)

        if periodic:
            if i != 0:
               for j in range(abs(i)):
                #W[j,N-abs(i)+j] = k/(2**(abs(i)-1))
                #W[N-1-j,abs(i)-1-j] = k/(2**(abs(i)-1))
                W[j,N-abs(i)+j] = 1
                W[N-1-j,abs(i)-1-j] = 1
        else:
            W[0:radius,:] = np.zeros((radius,N))
            W[-radius:,:] = np.zeros((radius,N))
                
    return W





def dist(N,i,j):
    
    return min(abs(i-j),N-abs(i-j))



def time_delay_matrix(N,alpha):
    N = int(N)
    A = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            A[i,j] = alpha*2*np.pi*dist(N,i,j)/(2*N)
    
    return A
    
    

## defines oscillator object with attributes phase ([0,2pi]) and frequency
class oscillator:
    def __init__(self,phase,frequency):
        self.phase = phase
        self.frequency = frequency



## creates population of oscillators with uniform random phases and frequencies drawn from a narrow normal distribution. 
## By default, uniform mean frequency (graditn = None). 
## set gradient = "linear","quadratic", or "exp" to introduce 
##      corresponding frequency gradient starting at freq_0 and ending at freq_final    
def create_population(N,freq_0,freq_std, gradient = None, delta_freq = 1):
    population = []

    for i in range(N):
        phase_i = 2*np.pi*np.random.rand()
        if gradient != None:
            if gradient == "linear":
                freq_i = freq_0+(i/(N-1))*delta_freq + np.random.normal(0,freq_std)
            if gradient == "quadratic":
                freq_i = freq_0 + (i/(N-1))**2*delta_freq+np.random.normal(0,freq_std)
            if gradient == "cubic":
                freq_i = freq_0 + (i/(N-1))**3*delta_freq+np.random.normal(0,freq_std)
            if gradient == "sqrt":
                freq_i = freq_0 + (i/(N-1))**0.5*delta_freq+np.random.normal(0,freq_std)
            if gradient == "exponential":
                freq_i = freq_0 + (np.exp(i)-1)/(np.exp(N-1)-1)*delta_freq +np.random.normal(0,freq_std)
            if gradient == "sine":
                freq_i = freq_0 + 6*np.sin(np.pi*i/N)        
        else:
            freq_i = np.random.normal(freq_0,freq_std)
        population.append(oscillator(phase_i,freq_i))

    return population



## introduces a total of num_defects defects into population.
## Each defect is implemented as an oscillator that is cut off from interacting with the rest

def introduce_defects(W,num_defects,loc_variability = False):
    
    N = np.shape(W)[0]
    indices = np.linspace(0,N-1,num_defects)
    
    for i in indices:
        
        if loc_variability:
            x = int(np.random.normal(float(i),2))
        else:
            x = int(i)
        
        W[:,x] = np.zeros(N)
        W[x,:] = np.zeros(N)
    
    return W



## single time step update to oscillator population according to Kuramoto model 


#def update_population(W,total_phases,thetas,frequencies,dt):
#    delta = dt*(frequencies+np.cos(thetas)*(W.dot(np.sin(thetas)))-np.sin(thetas)*(W.dot(np.cos(thetas))))
#    thetas += delta
#    thetas = np.mod(thetas,2*np.pi)
#    total_phases += delta
#    return (thetas, total_phases)

## f = dTheta/dt 
def f(W,k,frequencies,thetas):
    f = frequencies+k*np.imag(np.exp(-1j*thetas)*W.dot(np.exp(1j*thetas)))
    return f

## single time step update to oscillator population using a fourth order Runge-Kutta method (RK4)
def update_population_RK(W,k_temp,a,total_phases,thetas,frequencies,dt):
    
    k1 = dt*f(W,k,frequencies,thetas)
    
    k2 = dt*f(W,k,frequencies,thetas+0.5*k1)
    
    k3 = dt*f(W,k,frequencies,thetas+0.5*k2)
    
    k4 = dt*f(W,k,frequencies,thetas+k3)
    
    delta_theta = (k1+2*k2+2*k3+k4)/6
    
    thetas += delta_theta
    thetas = np.mod(thetas,2*np.pi)
    total_phases += delta_theta
    
    return (thetas, total_phases)


## models evolution of population for time T
def update_system(W,k_t,vel_t,total_phases,thetas,frequencies,T,dt,N):
    
    ## keeps track of population pattern (phases) in time 
    system_t  = np.zeros((int(T/dt),N))
    
    system_t_total = np.zeros((int(T/dt),N))
    
    for t in range(int(T/dt)):
        
        system_t[t,:] = thetas.flatten()
        system_t_total[t,:] = total_phases.flatten()
        
        if vel_t is None:
            a = None
        else:
            if t < 0.4*int(T/dt):
                a = 1
            else:
                a = vel_t[t]
        
        try:
            k_temp = k_t[t]
        except:
            k_temp = k_t
        
        thetas, total_phases = update_population_RK(W,k_temp,a,total_phases,thetas,frequencies,dt)
            
    
    return (system_t, system_t_total)






def simulate(N,k,k_dynamic,velocity,alpha,radius,periodic,defects,num_defects,freq_0,delta_freq,freq_std,gradient,T,dt):
    

    ## population of N oscillators 
    population = create_population(N,freq_0,freq_std,gradient = gradient,delta_freq = delta_freq)
    
    ## Nx1 vectors containing the phase and frequency of each oscillator 
    phases = np.array([[oscillator.phase for oscillator in population]]).T
    total_phases = np.array([[oscillator.phase for oscillator in population]]).T
    frequencies = np.array([[oscillator.frequency for oscillator in population]]).T
    
    
    if periodic == False:
        N = N + 2*radius
        left_phases = 2*np.pi*np.random.random_sample((radius,1))
        right_phases = 2*np.pi*np.random.random_sample((radius,1))
        phases = np.concatenate((left_phases,phases,right_phases))
        total_phases = np.ndarray.copy(phases)
        left_freqs = freq_0*np.ones((radius,1)) +  np.random.normal(0,freq_std,(radius,1))
        right_freqs = (freq_0+delta_freq)*np.ones((radius,1)) + np.random.normal(0,freq_std,(radius,1))    
        frequencies = np.concatenate((left_freqs,frequencies,right_freqs))
        
        
    ## Matrix of pair-wise oscillator couplings 
    W = interaction_matrix(N,periodic = periodic,radius = radius)
        
    ## Interaction matrix in sparse format
    W = scipy.sparse.csr_matrix(W)
    
    
    if k_dynamic:
        t = np.linspace(0,T,int(T/dt))
        k_t = k*(1-np.exp(-t/(0.2*T)))
    else:
        k_t = k
        
    ## time series of input velocities 
    if velocity:
        vel_t = 10*create_vel(T,dt) + 1
        np.save("trajectory.npy",vel_t)
        import matplotlib.pyplot as plt
        plt.plot(vel_t)
    else:
        vel_t = None
    

    ##updates all oscillators 
    system_t, system_t_total = update_system(W,k_t,vel_t,total_phases,phases,frequencies,T,dt,N)
    
    return system_t,system_t_total


""" ...........................Data Processing Functions................................."""


## Calculates numerical time derivative of oscillator phases at all times t 
def calc_eff_freq(system_t,dt):
    size = np.shape(system_t)
    eff_freqs_t = np.zeros((size[0]-2,size[1]))
    for t in range(1,size[0]-1):  
        delta = (system_t[t+1]-system_t[t-1])
        eff_freqs_t[t-1,:] = delta/(2*dt)
    return eff_freqs_t

    
""" ..........................Simulation................................. """

import sys

# N : number of oscillators
N = int(sys.argv[1])
# k : coupling constant
k = float(sys.argv[2])*0.001
# k_dynamic: set to True to make k a dynamic variable
k_dynamic = False
# velocity: set to True to introduce velocity inputs, False otherwise
velocity = False
# alpha: time delay constant
alpha = float(sys.argv[3])*0.01
# radius : radius of local interactions
radius = int(sys.argv[4])
# periodic : set to True for periodic topology, set to False for aperiodic topology
periodic = bool(sys.argv[5])
# defects : set to True to introduce sparse uniformly ditributed defects 
defects = bool(sys.argv[6])
# num_defects : specify number of defects to introduce
num_defects = int(sys.argv[7])
# freq_0 : initial center of frequency distribution
freq_0 = float(sys.argv[8])
# delta_freq : absolute change in frequency due to gradient. final freq = freq_0 + delta_freq
delta_freq = N*0.06 #float(sys.argv[9])
# freq_std : std of frequency distribution
freq_std = float(sys.argv[10])
# gradient : sets functional form of population frequency gradient. gradient one of {None,"linear","quadratic","exponential"}
gradient = str(sys.argv[11])
# T : simulation time length
T = int(sys.argv[12])
# dt : time step width
dt = float(sys.argv[13])



def fourier(eff_freqs,index,window_start,ff):
    s = eff_freqs[window_start:window_start+1000,index]
    signal = s - np.mean(s)
    f = np.abs(np.fft.fft(signal))
    return f[:ff]


def peak(s):
    h = np.mean(s)
    for i in range(len(s)):
        if s[i] == h:
            return i


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):	
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr 
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    a = plt.figure(dpi=200)
    plt.plot(x, y,color=color,marker="o",figure=a)
    plt.fill_between(x, ymax, ymin,color="red", alpha=alpha_fill)
    plt.xlabel("Oscillator")
    plt.ylabel("time-averaged dTheta/dt")
    plt.savefig(str("/om/user/facosta/Kuramoto/simulation/Figures/kuramoto_N="
    + str(N) + "_k=" + str(k) + "_alpha=" +str(alpha)+ "_R=" +
    str(radius) + "_delfreq=" +str(delta_freq) +'.png'))


ph = []

for r in range(96,200,4):
    
    radius = r

    start_time = time.time()

    system_t, system_t_total  = simulate(N,k,k_dynamic,velocity,alpha,radius,periodic,defects,num_defects,freq_0,delta_freq,freq_std,gradient,T,dt)

    total_time = start_time - time.time()

    #params = np.array([N,freq_0,T,dt])
    #np.savetxt(str('/om/user/facosta/Kuramoto/simulation/simulation_params/simulation_params_N='+str(N)+ '_k=' + str(k) + '_R=' + str(radius) + '.dat'),params)



    """ .............................Process Data......................................"""


    eff_freqs = calc_eff_freq(system_t_total,dt)

    ## keeps track of standard deviation of effective frequencies in time
    #freq_std_t = np.std(eff_freqs,axis=1)


    x = np.linspace(0,N,N)
    y = np.mean(eff_freqs[int(0.9*(T/dt)):,:],axis=0)
    #yerr = np.std(eff_freqs[int(0.9*(T/dt)):,:],axis=0)

    #errorfill(x,y,yerr,color = "black")

    g = np.diff(y)
    #plt.plot(g)

    from scipy import ndimage
    a = ndimage.gaussian_filter1d(g,sigma=5)
    #plt.plot(a)
    from scipy import signal
    h = signal.find_peaks(a,prominence=0.04)
    num_steps = len(h[0]) + 1

    ree = signal.find_peaks(np.diff(h[0]),threshold=radius)

    if len(ree[0]) > 0:
        num_steps = -1
        

    ph.append((k,radius,num_steps))

ph = np.array(ph)    

p_s = np.load("phase_space.npy")

if np.size(p_s) == 0:
    p_s = ph
else:
    p_s = np.concatenate((p_s,ph))

np.save("phase_space.npy",p_s)



    
##    ff = 60
##    m = np.zeros((N,ff))
##    for i in range(N):
##        m[i,:] = fourier(eff_freqs,i,40000,ff)
##
##    np.save('/om/user/facosta/Kuramoto/simulation/fourier/R='+str(radius)+'_k='+str(k)+'.npy',m)



    


