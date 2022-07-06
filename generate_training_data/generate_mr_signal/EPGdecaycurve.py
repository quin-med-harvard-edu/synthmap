"""This function replicates Prasloski's code for estimation of stimulated echo effects in CPMG signal 1-1 (from matlab)

Souce: /fileserver/motion/onur/MWF/code/prasloski/SEcorr_zip/EPGdecaycurve.m

"""

import numpy as np
import scipy.sparse as sps 

def test():

    # init vars
    ETL = 32 
    flip_angle = 180.
    TE = 9e-3
    T2 = 50e-3
    T1 = 1
    refcon = 180.0
    
    # for relaxmat 
    num_states,te,t2,t1 = ETL,TE,T2,T1
    
    # for flipmat 
    alpha,num_pulses,refcon = flip_angle*(np.pi/180),ETL,refcon

    # test whole function 
    decay_curve = EPGdecaycurve(ETL, flip_angle, TE, T2, T1, refcon)
    
def EPGdecaycurve(ETL,flip_angle,TE,T2,T1,refcon):
    # Computes the normalized echo decay curve for a MR spin echo sequence with the given parameters.
    #
    # ETL: Echo train length (number of echos)
    # flip_angle: Angle of refocusing pulses (degrees)
    # TE: Interecho time (seconds)
    # T2: Transverse relaxation time (seconds)
    # T1: Longitudinal relaxation time (seconds)
    # refcon: Value of Refocusing Pulse Control Angle

    # Initialize magnetization phase state vector (MPSV) and set all
    # magnetization in the F1 state.
    M=np.zeros((3*ETL,1))
    M[0,0]=np.exp(-(TE/2)/T2)*np.sin(flip_angle*(np.pi/180)/2)
    # M(1,1)=exp(-(TE/2)/T2)*sin(180*(pi/180)/2);
    # Compute relaxation matrix
    T_r=relaxmat(ETL,TE,T2,T1)
    # Initialize vector to track echo amplitude
    echo_amp=np.zeros((1,ETL))
    # Compute flip matrix
    [T_1,T_p]=flipmat(flip_angle*(np.pi/180),ETL,refcon)
    # Apply first refocusing pulse and get first echo amplitude
    #M[0:3]=T_1*M[0:3]
    M = M.astype('complex')
    M[0:3] = np.matmul(T_1, M[0:3])

    
    echo_amp[0,0]=np.abs(M[1,0]);
    echo_amp[0,0]=echo_amp[0,0]*np.exp(-(TE/2)/T2)
    # Apply relaxation matrix
    M=np.matmul(T_r,M)
    
    # Perform flip-relax sequence ETL-1 times
    for x in range(2,ETL+1):
        # Perform the flip
        M=np.matmul(T_p,M)
        # Record the magnitude of the population of F1* as the echo amplitude
        # and allow for relaxation
        echo_amp[0,x-1]=np.abs(M[1,0])*np.exp(-(TE/2)/T2)
        # Allow time evolution of magnetization between pulses
        M=np.matmul(T_r,M)
    
    decay_curve=echo_amp
    
    return decay_curve



def flipmat(alpha,num_pulses,refcon):
    # Computes the transition matrix that describes the effect of the refocusing
    # pulse on the magnetization phase state vector.

    # Compute the flip matrix as given by Hennig (1988), but corrected by Jones
    # (1997)
    T_1=np.array([[np.cos(alpha/2)**2,np.sin(alpha/2)**2,-1j*np.sin(alpha)],
                  [np.sin(alpha/2)**2,np.cos(alpha/2)**2,1j*np.sin(alpha)],
                  [-0.5j*np.sin(alpha),0.5j*np.sin(alpha),np.cos(alpha)]])
    
    alpha2=alpha*refcon/180
    
    T_2=np.array([[np.cos(alpha2/2)**2,np.sin(alpha2/2)**2,-1j*np.sin(alpha2)],
                  [np.sin(alpha2/2)**2,np.cos(alpha2/2)**2,1j*np.sin(alpha2)],
                  [-0.5j*np.sin(alpha2),0.5j*np.sin(alpha2),np.cos(alpha2)]])
    
    # Create a block matrix with T_1 on the diagonal and zeros elsewhere
    #T_p=spalloc(3*num_pulses,3*num_pulses,9*num_pulses);
    T_p = np.zeros((3*num_pulses, 3*num_pulses), dtype = complex)
    #T_p = sps.csr_matrix(T_p,dtype=np.float) 
    #T_p.eliminate_zeros()
    #T_p = T_p.todense()
    
    
    for x in range(1, num_pulses+1):        
        #T_p[3*x-3:3*x-1,3*x-3:3*x-1] = T_2
        T_p[3*x-3:3*x,3*x-3:3*x] = T_2
    np.where(T_p!=0)
    return [T_1,T_p]


def relaxmat(num_states,te,t2,t1):
    
    # Computes the relaxation matrix that describes the time evolution of the
    # magnetization phase state vector after each refocusing pulse.

    # Create a matrix description of the time evolution as described by
    # Hennig (1988)

    #T_r=np.zeros((3*num_states,3*num_states), dtype=complex)
    T_r=np.zeros((3*num_states,3*num_states))
    
    # F1* --> F1
    T_r[0,1]=np.exp(-te/t2)
    # F(n)* --> F(n-1)*
    for x in range(1,num_states):
        T_r[3*x-2,3*x+1]=np.exp(-te/t2)
    T_r[np.where(T_r!=0)]
    
    # F(n) --> F(n+1)
    for x in range(1, num_states):
        T_r[3*x,3*x-3]=np.exp(-te/t2)
    
    # Z(n) --> Z(n)
    for x in range(1, num_states+1):
        T_r[3*x-1,3*x-1]=np.exp(-te/t1)
    

    #T_r=sparse(T_r);
    #T_r = sps.csr_matrix(T_r,dtype=np.float) 
    #T_r.eliminate_zeros()
    #T_r = T_r.todense()
    
    
    return T_r 