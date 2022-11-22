#!/usr/bin/env python
# coding: utf-8

# """
# Created on Tue Apr  5 15:53:11 2016
# Last change: August 10, 2018
# 
# This program runs the LMS algorithm in Python
# Input signal: AR(1) process with a white Gaussian driving noise
# Structure: System identification with an input delay line
# Unknown system: Nonstationary using the random walk model
# 
# This program runs function classical_model in the GPU.
# The function Monte_Carlo runs in the CPU, which is faster.
# 
# The whole program runs about 2 times faster than LM3_numpy2.ipynb, which runs completely in the CPU (tested for N=512).
#
# In this version I have optimized the updating of the input vector by eliminating 
# the flip-left-right operation at each iteration, replacing it by the right shif of the vector and
# addition of the new input sample. The MC simulatio became twice as fast for N=512.
# 
# Para LMS3_cupy6.py, modify LMS3_cupy5.py to try to move the MC simulation to the GPU
#
#
# @author: bermudez
# """

# In[1]:


# Importing the modules necessary for scientific calculations
import numpy as np              # module for working with arrays
import cupy as cp
import scipy.linalg as linalg   # modulee for linear algebra
#import cupyx.scipy.linalg as cplinalg
import matplotlib.pyplot as plt # module for plotting
#import myfunctions as mf        # module for my own functions


# In[2]:


# Setting time counter
import time
from datetime import datetime
start_time = datetime.now()


# ### Function definitions

# In[3]:


# Classical LMS model for Gaussian inputs (Eq. 10.4.28, Manolakis' book)
def classical_model(K0, R, varz, varq, mu, N, iterations):
    msdt = cp.zeros(iterations)
    emset = cp.zeros(iterations)
    K = K0
    msdt[0] = cp.trace(K)
    emset[0] = cp.trace(R @ K)
    for m in cp.arange(1,iterations):
        K = K - mu * (R @ K + K @ R) + mu**2 * (cp.trace(R @ K) * R + 2 * R @ K @ R) + mu**2 * varz * R + varq * cp.eye(N)
        msdt[m] = cp.trace(K)
        emset[m] = cp.trace(R @ K)
        
    return msdt, emset 


# In[4]:


# Monte Carlo simulation
def Monte_Carlo(runs, SystemResponse, alpha, varx, varz, varq, N, iterations):
    mse = np.zeros(iterations)            # Mean square error
    emse = np.zeros(iterations)           # Excess mean square error
    msd = np.zeros(iterations)            # Mean square deviation
    for r in np.arange(runs):                     # loop for realizations
        SystemResponsen = SystemResponse          # Updates the channel response
        W = np.zeros(N)                           # initializes W(0)

        # Input driving noise generation
        X0 = np.sqrt((1-alpha**2)*varx)*np.random.normal(0,1,size=iterations+N+1)
        X1 = np.zeros(iterations+N+1)            # initializes vector of input samples
        X1[0] = X0[0]
        for k in np.arange(1,iterations+N+1):    # AR(1) model
            X1[k] = alpha*X1[k-1] + X0[k]


        # Generation of the measurement noise sequence
        noise = np.sqrt(varz)*np.random.randn(iterations+N)

        # Generation of initial input vector
        X = X1[0:N]                         # generates input vector
        X = X[::-1]                           # flips the vector around
        
        # Adaptive algorithm
        for n in np.arange(iterations):
            z = noise[n]                          # generates additive noise sample
            d = X @ SystemResponsen + z           # evaluates desired signal
            y = X @ W                             # evaluates adaptive filter output
            e = d - y                             # evaluates error signal
            mse[n] = mse[n] + e**2                # accumulates e²(n)
            emse[n] = emse[n] + (e-z)**2          # accumulates (e(n)-z(n))²
            V = W - SystemResponsen               # evaluates weight error vector
            msd[n] = msd[n] + V @ V               # evaluates MSD(n)
            # Updating the adaptive weights
            W = W + mu * e * X
            # Update of unknown system response
            SystemResponsen = SystemResponsen + np.sqrt(varq) * np.random.randn(N)
            # Update of the input vector
            X = np.concatenate((np.array([X1[n+N]]), X[:-1]))
        
    return mse, emse, msd  

# ### Simulation parameters

# In[5]:


N = 512                      # number of adaptive coefficients
iterations = 50000          # number of iterations
runs = 100                    # number of Monte Carlo realizations
mu = 5e-4                  # step-size
varx = 1.                    # variance of the input signal
varz = 1e-6                  # variance of the additive measurement noise
alpha = 0.4                  # correlation parameter for the AR(1) input model
               

# Markov channel fluctuations (choose one)
eta = 0.                         # degree of nonsationarity due to channel changes
varq = (varz*(eta**2))/(N*varx)   # Markov  Channel fluctuations


# ### Define system response

# In[6]:


#SystemResponse = mf.raiscos(N,0.,5,0.8)
SystemResponse = np.array(np.random.randn(N))

# Normalization of the system response
SystemResponse = SystemResponse/np.sqrt(SystemResponse.dot(SystemResponse))

#cp.save('data/response.npy',SystemResponse) # do it once for all simulations
#SystemResponse = cp.load('data/responseraiscos.npy') # use this once the response is saved
#SystemResponse = cp.load('data/responsedecay.npy') # use this once the response is saved


# ### Input autocorrelation matrix

# In[7]:


# Theoretical input autocorrelation matrix 

liner = np.array(varx*(alpha**(np.arange(N))))
# Input autocorrelation matrix
R = linalg.toeplitz(liner)

# Calculation of eigenvectors and eingenvalues
#   Each element of la is one eigenvalue (ordered by increasing magnitude)
#   Each column of S is an eigenvector in the same order
la, S = np.linalg.eigh(R)
#print("eigenvalues:",cp.abs(la), '\n')
#print("eigenvectors: ", "\n", S, '\n')
spread = np.abs(np.max(la))/np.abs(cp.min(la))
print('eigenvalue spread: ', spread)

# Input eigenvalue matrix
Lambda0 = np.diag(np.abs(la))      
#print("Lambda0 = ", '\n', Lambda0, '\n')


#  ### Simulations

# In[8]:


# Vector initializations
mse = np.zeros(iterations)            # Mean square error
emse = np.zeros(iterations)           # Excess mean square error
msd = np.zeros(iterations)            # Mean square deviation
msdtheo = cp.zeros(iterations)        # Theoretical MSD(n)
emsetheo = cp.zeros(iterations)        # Theoretical EMSE(n)
SystemResponsep = cp.asarray(SystemResponse)
K0 = cp.outer(SystemResponsep,SystemResponsep)  # Matrix K(0)


# In[9]:


# Call function classical_model
#classical_params_real = cp.array([varz, varq, mu])
#classical_params_int = cp.array([N, iterations], dtype=int32)
st_time = time.time()
msdtheo, emsetheo = classical_model(K0, cp.asarray(R), varz, varq, mu, N, iterations)
#msdtheo, emsetheo = classical_model(K0, cp.asarray(R), classical_params)
#print('Call time: ', time.time()-st_time)
msdtheo = 10 * cp.log10(msdtheo)
emsetheo = 10 * cp.log10(emsetheo)
print('Call time: ', time.time()-st_time)


# In[10]:


# Call function Monte Carlo realizations
st_time = time.time()
mse, emse, msd = Monte_Carlo(runs, SystemResponse, alpha, varx, varz, varq, N, iterations)
print('Call time: ', time.time()-st_time)


# ### Averaging

# In[11]:


mse = mse/runs                                # evaluates MSE
emse = emse/runs                              # evaluates EMSE
msd = msd/runs                                # evaluates MSD
msedb = 10 * np.log10(mse)                    # MSE in dB
emsedb = 10 * np.log10(emse)                  # EMSE in dB
msddb = 10 * np.log10(msd)                    # MSD in dB


# ### Plotting the results

# In[12]:


# Plotting the results
plt.figure()
t = np.arange(iterations)
plt.plot(t,emsedb,label='simulations'), plt.grid()    
plt.plot(t,emsetheo.get(),label='theory')  
#plt.plot(t,10*np.log10(Jex),label='theory model'),plt.grid()
plt.legend()                           # adds legend with labels defined above
plt.title('Excess Mean Squared Error', fontsize=18)
plt.xlabel('iterations', fontsize=14)
plt.ylabel('EMSE (dB)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax = plt.gca()                    # grabs current axes and names them ax
# Setting axes properties
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # scientific number style
ax.xaxis.offsetText.set_fontsize(14)                        # offset text font size
#plt.savefig('testfig.pdf',bbox_inches='tight')
#plt.show()

plt.figure()
plt.plot(t,msddb,label='simulations'), plt.grid()    
plt.plot(t,msdtheo.get(),label='theory')  
#plt.plot(t,10*np.log10(Jex),label='theory model'),plt.grid()
plt.legend()                           # adds legend with labels defined above
plt.title('Mean Square Deviation', fontsize=18)
plt.xlabel('iterations', fontsize=14)
plt.ylabel('MSD (dB)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax = plt.gca()                    # grabs current axes and names them ax
# Setting axes properties
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # scientific number style
ax.xaxis.offsetText.set_fontsize(14)                        # offset text font size
#plt.savefig('testfig.pdf',bbox_inches='tight')
plt.show()


# ### Print execution time

# In[13]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[ ]:




