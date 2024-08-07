# Implementing the projection function

Author: Hao Mack Yang Li

The goal of this demonstration is to display the implementation of nonunitary operators on qubits using unitary block-encoded matrices in _Qmod_. Also, the starting qubit has a 30% probability of being measured in the zero state $|0\rangle$.

In this demonstration, the nonunitary operator $A$ is the projector to zero state $$A = |0\rangle\langle0| = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$

This operator collapses any single-qubit state $\alpha|0\rangle + \beta|0\rangle$ to $|0\rangle$.

## Linear Combination of Unitaries

The projector is an equal sum of the Pauli matrices $I$ and $Z$. Therefore, the linear combination of unitaries is expressed as $$\frac{1}{2}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} + \frac{1}{2}\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} = \begin{bmatrix}\frac{1}{2} + \frac{1}{2} & 0 \\ 0 & \frac{1}{2} - \frac{1}{2} \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$

Let the sequences $\{\alpha_k\}_k = 0.5, 0.5$ and $\{U_k\}_k = I, Z$. This sequence is used for the _Qmod_ implementation.

## Qmod implementation

The implementation contains two steps: (1) Preparing the selector The _Qmod_ code uses the _within_/_apply_ and _control_ keywords to simplify the second step.

```
qfunc main(output psi: qbit, output ctrl: qbit) {
  // Allocate and rotate input qubit
  allocate(1, psi);
  RY(2*acos(sqrt(0.3)), psi);

  // Apply the nonunitary gate by applying extended unitary gate on two qubits
  allocate(1, ctrl);
  within {
    inplace_prepare_state([0.5, 0.5], 0.001, ctrl);
  } apply {
    control (ctrl == 0) {
      I(psi);
    }
    control (ctrl == 1) {
      Z(psi);
    }
  }
}
```

## Circuit

An RX circuit is used to transform the $|0\rangle$ to the state $\sqrt{0.3}|0\rangle + \sqrt{0.7}|1\rangle$. The state preparation gate, with $\alpha_0 = \alpha_1$, is simply the Hadamard gate. The inverse preparation gate, likewise, is also the Hadamard gate.

### Diagram

![Circuit](HaoMackYang_Lee_WQ6-10.png)

### Simplified circuit

The circuit can be further simplified by removing the controlled-I operator.

```
qfunc main(output psi: qbit, output ctrl: qbit) {
  allocate(1, psi);
  allocate(1, ctrl);
  RY(2*acos(sqrt(0.3)), psi);
  H(ctrl);
  CZ(ctrl, psi);
  H(ctrl);
}
```

![Circuit](HaoMackYang_Lee_WQ6-10a.png)

## Results

Theoretically, the control qubit should remain at the $|0\rangle$ state, while the target qubit has collapsed to the $|0\rangle$ state. After measuring both qubits, the theoretical probability that the measurement will return a $|0\rangle|0\rangle$ is unity.

In our run, about 30% of the shots end up on the $|0\rangle|0\rangle$ state. 
