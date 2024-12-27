# Quantum search with a non-unitary gate
The simulation code for the paper, An alternative non-unitary implementation for the quantum search algorithm, A. Daskin, 2024
https://arxiv.org/abs/2412.16514
latest version on researchgate: http://dx.doi.org/10.13140/RG.2.2.25468.24960

The code basically applies the left most matrix as an approximation of the right most matrix:
```math
\left( \begin{matrix}-\frac{n-2}{n}&1\\1&-1\end{matrix} \right) \approx\left( \begin{matrix}-\frac{poly(n)-2}{poly(n)}&1\\1&-1\end{matrix} \right) \approx \left( \begin{matrix}-\frac{N-2}{N}&1\\1&-1\end{matrix} \right) \approx \left( \begin{matrix}-1&1\\1&-1\end{matrix} \right)
```
   - you can use $n-1$ or any other value too. You can also parameterize this...
   - Note that this is kind of related to the precision of the machine.

This is applied to the first qubit.
- This constructs the vector similar to the final state of the Grover search algorithm. 
- And when this state is collapsed on to the first part by a measurement on the first qubit, the remaining state (the collapsed)  is automatically normalized. 

    - This normalized state is similar to the result of the Gram-Schmidt orthogonalization process applied to the the initial superposition and the marked vector.

## Unitary implementation
- Unitary implementation of this circuit is done through the square root of the matrix. This is shown in the paper.
- unitarycircuit.py simulates this circuit with or without AA. The number of repetitions are set to n: you can adjust this and the parameter a.
