import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def create_block_encoding(a):
    M = np.array([[a, -1], [-1, a]])
    kappa = a + 1
    C = M / kappa
    
    # Eigen decomposition for matrix square root
    eigvals, eigvecs = np.linalg.eigh(C)
    S_eigvals = np.sqrt(np.maximum(1 - eigvals**2, 0))  # Ensure real values
    S = eigvecs @ np.diag(S_eigvals) @ eigvecs.T
    
    # Construct unitary block encoding
    top = np.hstack((C, -S))
    bottom = np.hstack((S, C))
    return np.vstack((top, bottom))

def circuit(n, a, marked_state, wires):
    # Prepare initial states
    qml.Hadamard(wires=wires[1])
    for i in wires[2:]:
        qml.Hadamard(wires=i)
    
    # Apply controlled oracle - mark when control=1 and data=marked_state
    diag = np.ones(2**(n+2))
    index0 = (0 << (n+1)) | (1 << n) | marked_state  # ancilla=0, control=1, data=marked
    index1 = (1 << (n+1)) | (1 << n) | marked_state  # ancilla=1, control=1, data=marked
    diag[index0] = -1
    diag[index1] = -1
    qml.DiagonalQubitUnitary(diag, wires=wires)
    
    # Apply block encoding to ancilla and control qubits
    Uc = create_block_encoding(a)
    qml.QubitUnitary(Uc, wires=wires[:2])

def simulate(n, a, marked_state):
    num_wires = n + 2
    wires = list(range(num_wires))
    dev = qml.device("default.qubit", wires=wires)
    
    @qml.qnode(dev)
    def qnode():
        circuit(n, a, marked_state, wires)
        return qml.state()
    
    state = qnode()
    probs = np.abs(state)**2
    
    # Calculate probability of ancilla=0
    p0 = np.sum(probs[:2**(num_wires-1)])
    
    # Calculate joint probability (ancilla=0 and data=marked_state)
    p_joint = 0.0
    for c in [0, 1]:  # Sum over control qubit states
        idx = (0 << (n+1)) | (c << n) | marked_state
        p_joint += probs[idx]
    
    p_marked_given_0 = p_joint / p0 if p0 > 0 else 0
    return p0, p_marked_given_0

# Parameters
n_values = list(range(2, 14))  # Number of data qubits
a_values = [(n**2-1)/n**2 for n in n_values]
results = []

for i, n in enumerate(n_values):
    marked_state = np.random.randint(0, 2**n)
    a = a_values[i]
    p0, p_marked = simulate(n, a, marked_state)
    results.append((n, p0, p_marked))

# Plotting
n_vals = [r[0] for r in results]
p0_vals = [r[1] for r in results]
p_marked_vals = [r[2] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(n_vals, p0_vals, 'o-', label='$P_0$ (ancilla=0)')
plt.plot(n_vals, p_marked_vals, 's-', label='$P$(marked|ancilla=0)')
plt.plot(n_vals, [1/n for n in n_vals], '^--', label='$1/n$')
plt.plot(n_vals, [1/np.sqrt(2**n) for n in n_vals], 'x--', label='$1/\sqrt{N}$')
plt.plot( n_vals, a_values,'d--', label='$a$')
plt.xlabel('Number of Data Qubits (n)')
plt.ylabel('Probability')
plt.legend()
plt.title('Unitary Search with Ancilla (without AA for ancilla=0)')
plt.xticks(n_vals)
plt.grid(True)
plt.show()