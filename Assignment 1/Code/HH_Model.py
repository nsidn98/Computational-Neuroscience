#THIS PROGRAM DEMONSTRATES HODGKIN HUXLEY MODEL IN CURRENT CLAMP EXPERIMENTS AND SHOWS ACTION POTENTIAL PROPAGATION
#Time is in secs, voltage in mvs, conductances in m mho/mm^2, capacitance in uF/mm^2

# threshold value of current is 0.0223


import numpy as np
import matplotlib.pyplot as plt


g_K_max=.36 # max conductance of K channel
V_K=-77 #voltage of K channel
g_Na_max=1.20 #max conductance of Na channel
V_Na=50 #voltage of Na channel
g_l=0.003 #conductance of combined gates
v_l=-54.387 #voltageof combined channel
cm=.01

dt=0.01 #0.01 ms
niter=10000
t=np.array([i for i in range(niter)])
I_app=ImpCur*np.ones(niter)
V=-64.9964 #base voltage
m=0.0530
h=0.5960
n=0.3177


#### to store the values
g_Na_hist=np.zeros(niter)
g_K_hist=np.zeros(niter)
V_hist=np.zeros(niter)
m_hist=np.zeros(niter)
h_hist=np.zeros(niter)
n_hist=np.zeros(niter)

for i in range(niter):
    g_Na = g_Na_max*(m**3)*h
    g_K = g_K_max*(n**4)
    g_total = g_Na+g_K+g_l
    V_inf = ((g_Na*V_Na+g_K*V_K+g_l*V_l)+I_app[i])/g_total
    tau_v = cm/g_total
    V = V_inf+(V- V_inf)*np.exp(-dt/tau_v)
    alpha_m = 0.1*(V+40)/(1-np.exp(-(V+40)/10))
    beta_m = 4*np.exp(-0.0556*(V+65))
    alpha_n = 0.01*(V+55)/(1-np.exp(-(V+55)/10))
    beta_n = 0.125*np.exp(-(V+65)/80)
    alpha_h = 0.07*np.exp(-0.05*(V+65))
    beta_h = 1/(1+np.exp(-0.1*(V+35)))
    tau_m = 1/(alpha_m+beta_m)
    tau_h = 1/(alpha_h+beta_h)
    tau_n = 1/(alpha_n+beta_n)
    m_inf = alpha_m*tau_m
    h_inf = alpha_h*tau_h
    n_inf = alpha_n*tau_n
    m=m_inf+(m-m_inf)*exp(-dt/tau_m)
    h=h_inf+(h-h_inf)*exp(-dt/tau_h)
    n=n_inf+(n-n_inf)*exp(-dt/tau_n)
    V_hist[i]=V
    m_hist[i]=m
    h_hist[i]=h
    n_hist[i]=n
    
plt.plot(t,V_hist)
plt.show()


    











