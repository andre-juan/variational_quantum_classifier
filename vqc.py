# imports

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

###########################################################################################################
###########################################################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
###########################################################################################################
###########################################################################################################

# functions for the circuit architecture

def initialize_circuit(Nq, Nc):
    '''
    initializes a circuit with Nq quantum registers and Nc classical registers
    '''
    
    # quantum and classical registers
    qr = QuantumRegister(Nq, "q")
    cr = ClassicalRegister(Nc, "c")

    # quantum circuit
    qc = QuantumCircuit(qr, cr)

    return qc

#########################################################################################
#########################################################################################

def amplitude_feature_map(qc, X_k, rep_fm=1):
    '''
    encodes classical data (in the array X) into the quantum circuit initial state.
    each component of the X array is encoded in the respective quantum register.
    the encoding is achieved via a Rx rotation, whose angle is the respective X component
    feature map repeated a number "rep_fm" of times
    '''
    
    for _ in range(rep_fm):
        
        # data encoding with Rx rotations
        for i, phi_ik in enumerate(X_k):

            qc.rx(phi_ik, i)
        
    return qc

#########################################################################################
#########################################################################################

def ZZ_feature_map(qc, X_k, rep_fm=1):
    '''
    Pauli-ZZ expansion circuit as a feature map.
    implements complex phases P and linear entanglement between qubits.
    the phase angle is the respective X component.
    for the general pauli feature map: https://qiskit.org/documentation/stubs/qiskit.circuit.library.PauliFeatureMap.html
    for the particular zz feature map: https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html
    the layer is applied a "rep_fm" number of times.
    '''
    
    for _ in range(rep_fm):
        
        for i in range(qc.num_qubits):
            
            qc.h(i)
            
            # phi_ik = X_k[i]
            qc.p(2*X_k[i], i)
        
        for i in range(qc.num_qubits-1):
            
            qc.cnot(i, i+1)
            
            qc.p(2*(np.pi - X_k[i])*(np.pi - X_k[i+1]), i+1)
            
            qc.cnot(i, i+1)
        
    return qc

#########################################################################################
#########################################################################################

def variational_circuit(qc, theta, rep_var=1):
    '''
    implements the variational portion of the quantum cirrcuit.
    the particular architecture will be: CNOTs between the registers;
    followed by parameterized Ry rotations (parameters defined by the argument "theta");
    layer repeated a number "rep_var" of times
    '''
    
    for _ in range(rep_var):
        
        N = qc.num_qubits
        
        # CNOTs to induce entanglement
        for i in range(N-1):

            qc.cnot(i, i+1)

        # final CNOT, controlled by the last qubit and targeted on the first qubit
        qc.cnot(-1, 0)

        # parameterized rotations
        for i in range(N):

            qc.ry(theta[i], i)
        
    return qc
        
#########################################################################################
#########################################################################################

def measurement(qc):
    '''
    measure the first quantum register into the classical register
    '''
    
    qc.measure(0, -1)
    
    return qc

###########################################################################################################
###########################################################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
###########################################################################################################
###########################################################################################################

# full function to construct the circuit

def build_circuit(Nq, Nc, X, theta, k, feature_map="amplitude", rep_fm=1, rep_var=1, show_circuit=True):
    '''
    construct and return the quantum circuit for the variational quantum classifier,
    with specified parameters
    '''

    qc = initialize_circuit(Nq, Nc)
    
    if feature_map == "amplitude":
        amplitude_feature_map(qc, X.iloc[k], rep_fm=rep_fm)
    
    elif feature_map == "ZZ":
        ZZ_feature_map(qc, X.iloc[k], rep_fm=rep_fm)
    
    variational_circuit(qc, theta, rep_var=rep_var)
    
    measurement(qc)
    
    if show_circuit:
        
        show_figure(qc.draw("mpl"))
        
    return qc



###########################################################################################################
###########################################################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
###########################################################################################################
###########################################################################################################

# functions for the circuit execution

def run_circuit(qc, simulator=True, backend_name = "qasm_simulator", n_runs=1e3, seed_simulator=None):
    '''
    executes the quantum circuit in the specified backend (in the moment, it's just a simulator)
    TODO: allow for actual hardware
    the number of executions can also be specified.
    returns the results of the execution.
    '''
    
    if simulator:
        
        # import provider for simulator and define the simulator backend
        backend = Aer.get_backend(backend_name)
        
    else:
        
        pass
        # # todo: quantum hardware
        # backend = ...
        
    # defining the job and sending it to execute in the defined backend
    # we will run the circuit a n_runs number of times
    job = execute(qc, backend, shots=n_runs, seed_simulator=seed_simulator)
    
    # getting the results of the job execution
    results = job.result()
    
    return results

#########################################################################################
#########################################################################################

def show_figure(fig):
    '''
    auxiliar function to display plot 
    even if it's not the last command of the cell
    from: https://github.com/Qiskit/qiskit-terra/issues/1682
    '''
    
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)

#########################################################################################
#########################################################################################

def final_answer(results, thresh=0.5, n_runs=1e3, visualize_results=False):
    '''
    calculate the probability of the observation belonging to class 1
    i.e., like the logit, calculation of P(y=1|x). 
    the actual class prediction is also calculated. both are returned.
    an optional visualization of the execution results as a histogram is also available
    '''

    # getting the resulting counts (of measurements in the classical register),
    counts = results.get_counts()
    
    if visualize_results:
        # ploting the counts as a histogram
        show_figure(plot_histogram(counts, title="Results"))
        
    # p(y=1 | x) - probability of observation belonging to class 1
    if "1" in counts.keys():
        p_y = counts["1"]/n_runs
    else:
        p_y = 0
    
    # actual prediction, according to the specified threshold
    y_pred = 1 if p_y > thresh else 0
    
    return p_y, y_pred


###########################################################################################################
###########################################################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
###########################################################################################################
###########################################################################################################


# full vqc function (takes the quantum circuit as input and executes it)

def vqc(qc, simulator=True, backend_name="qasm_simulator", n_runs=1e3,
        thresh=0.5, visualize_results=False, seed_simulator=None):
    '''
    this function integrates all the functions above in a single structure
    and returns p_y
    the only input is the integer k, used as an index for the chosen observation
    returns the final answer (probability and prediction)
    '''
    
    results = run_circuit(qc, seed_simulator=seed_simulator, n_runs=n_runs)

    return final_answer(results, visualize_results=visualize_results)


###########################################################################################################
###########################################################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
###########################################################################################################
###########################################################################################################

def loss_SE(p_y_k, y, k):
    '''
    computes the squared error (SE) loss
    '''
    
    return (p_y_k - y.iloc[k])**2

#########################################################################################
#########################################################################################

def loss_AE(p_y_k, y, k):
    '''
    computes the absolute error (AE) loss
    '''
    
    return np.abs(p_y_k - y.iloc[k])

#########################################################################################
#########################################################################################

def loss_BCE(p_y_k, y, k):
    '''
    computes the binary cross-etropy (BCE) loss
    '''
    
    return -(y.iloc[k]*np.log(p_y_k) + (1-y.iloc[k])*np.log(1-p_y_k))


###########################################################################################################
###########################################################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
###########################################################################################################
###########################################################################################################

def gradient(N, X, y, k, p_y_k, theta, loss_func="SE", delta=5e-3, n_runs=1e3, 
             seed_simulator=None, feature_map="amplitude", rep_fm=1, rep_var=1):   
    '''
    computes the directional derivative using the finite difference method
    '''
    
    theta_list = []
    for i in range(len(theta)):

        theta_list.append(theta.copy())
        theta_list[-1][i] = theta_list[-1][i] + delta
    
    del_theta = []
    
    for theta_pd in theta_list:
        
        # below, pd = "plus delta"
        
        qc_pd = build_circuit(N, 1, X, theta_pd, k, show_circuit=False, 
                              feature_map=feature_map, 
                              rep_fm=rep_fm, rep_var=rep_var)
        
        p_y_k_pd, _ = vqc(qc_pd, seed_simulator=seed_simulator, n_runs=n_runs)

        if loss_func == "SE":
        
            loss_theta = loss_SE(p_y_k, y, k)
            loss_theta_pd = loss_SE(p_y_k_pd, y, k)
            
        elif loss_func == "AE":
        
            loss_theta = loss_AE(p_y_k, y, k)
            loss_theta_pd = loss_AE(p_y_k_pd, y, k)
            
        elif loss_func == "BCE":
        
            loss_theta = loss_BCE(p_y_k, y, k)
            loss_theta_pd = loss_BCE(p_y_k_pd, y, k)

        derivative = (loss_theta_pd - loss_theta)/delta

        del_theta.append(derivative)
        
    return np.array(del_theta)

#########################################################################################
#########################################################################################

def train_accuracy(y_pred_k, y, k):
    '''
    auxiliar function to calculate train accuracy
    '''
    
    return 1 if y_pred_k == y.iloc[k] else 0

#########################################################################################
#########################################################################################
    
def stochastic_gd(N, X, y, theta, delta=5e-3, loss_func="SE", lr=5e-2, n_runs=1e3, num_epochs=50,
                  show_progress=True, seed_simulator=None, feature_map="amplitude", rep_fm=1, rep_var=1):
    '''
    this is the only function to be called for the training of the vqc using sgd
    
    parameters choices:
    
    - loss_func: available loss functions to be optimized: "SE", "BCE", "AE"
    - feature_map: available feature maps: "amplitude", "ZZ".
    '''
    
    loss_each_epoch = []
    accuracy_each_epoch = []
    
    if show_progress:
        print("{:<20s} {:<20s} {:<20s}".format("Epoch", "Loss", "Training Accuracy"))
    
    for n in range(num_epochs):
        
        loss_inter_epoch = []
        accuracy_inter_epoch = []
        
        for k in range(X.shape[0]):
        
            qc = build_circuit(N, 1, X, theta, k, show_circuit=False, 
                               feature_map=feature_map, 
                               rep_fm=rep_fm, rep_var=rep_var)

            p_y_k, y_pred_k = vqc(qc, seed_simulator=seed_simulator, n_runs=n_runs)
            
            if loss_func == "SE":
            
                loss_inter_epoch.append(loss_SE(p_y_k, y, k))
                
            elif loss_func == "AE":
                
                loss_inter_epoch.append(loss_AE(p_y_k, y, k))
                
            elif loss_func == "BCE":
                
                loss_inter_epoch.append(loss_BCE(p_y_k, y, k))
            
            accuracy_inter_epoch.append(train_accuracy(y_pred_k, y, k))
            
            theta = theta - lr * gradient(N, X, y, k, p_y_k, theta=theta, 
                                          seed_simulator=seed_simulator, loss_func=loss_func,
                                          delta=delta, n_runs=n_runs, feature_map=feature_map,
                                          rep_fm=rep_fm, rep_var=rep_var)
    
        loss_each_epoch.append(np.mean(loss_inter_epoch)) 
        accuracy_each_epoch.append(np.mean(accuracy_inter_epoch)) 
        
        if show_progress:
            print("{:<20d} {:<20.5f} {:<20.5f}".format(n+1, loss_each_epoch[-1], accuracy_each_epoch[-1]))
            
    return loss_each_epoch, accuracy_each_epoch, theta