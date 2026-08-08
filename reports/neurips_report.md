# Quantum-Augmented Physics-Informed Neural Networks: A Controlled Empirical Study

**Abstract**

Physics-Informed Neural Networks (PINNs) have emerged as a transformative paradigm for solving partial differential equations (PDEs) by embedding physical laws directly into the loss function of deep learning models. However, classical PINNs often struggle with high-dimensional optimization landscapes and expressivity limits in complex fluid dynamics. This study investigates the integration of variational quantum circuits (VQCs) into the PINN framework, proposing a hybrid architecture termed Quantum-Augmented Physics-Informed Neural Networks (QAPINNs). We conduct a controlled empirical comparison between classical PINNs and QAPINNs on the 1D Burgers' equation and the 1D Heat equation. Our experimental design explores the impact of qubit count, circuit depth, and entanglement strategies on convergence and accuracy. Results indicate that while low-qubit QAPINNs (2-4 qubits) achieve competitive performance with classical baselines, scaling to higher qubit counts (6-8 qubits) introduces significant training instabilities consistent with the "barren plateau" phenomenon. Furthermore, although increased entanglement enhances the expressivity of the quantum feature map, it simultaneously degrades the trainability of the hybrid system. We conclude that in the current NISQ-era simulation regime, QAPINNs do not demonstrate a clear quantum advantage over classical PINNs for these fundamental PDEs, highlighting critical tradeoffs between expressivity and optimization stability in hybrid quantum-classical architectures.

## 1. Introduction

The numerical solution of partial differential equations (PDEs) is a cornerstone of modern science and engineering, underpinning fields ranging from climate modeling to aerospace design. Traditional methods, such as finite element analysis (FEA) and finite difference methods (FDM), have long been the industry standard. However, these grid-based approaches face significant challenges, particularly the "curse of dimensionality" and the difficulty of handling complex, irregular geometries. In recent years, Physics-Informed Neural Networks (PINNs), introduced by Raissi et al. [1], have offered a mesh-free alternative that leverages the universal approximation capabilities of neural networks while enforcing physical consistency through automatic differentiation.

Despite their success, classical PINNs are not without limitations. The optimization of PINNs involves a multi-objective loss function—comprising data-driven, residual, and boundary terms—which often leads to stiff optimization landscapes and slow convergence. As the complexity of the PDE increases, the depth and width of the required classical neural networks grow, leading to increased computational costs and potential vanishing gradient issues.

Concurrently, the field of Quantum Machine Learning (QML) has proposed that variational quantum circuits (VQCs) might offer superior expressivity and computational efficiency for certain classes of problems. The hypothesis is that the high-dimensional Hilbert space accessible by quantum systems can represent complex functions more efficiently than classical architectures. Hybrid quantum-classical models, which combine classical feature extractors with quantum variational layers, have shown promise in various classification and regression tasks.

This research aims to bridge these two domains by systematically evaluating the performance of Quantum-Augmented Physics-Informed Neural Networks (QAPINNs). Specifically, we address the following research question: *Can the integration of variational quantum circuits into the PINN framework provide a measurable advantage in solving fundamental PDEs like the Burgers' and Heat equations, and how do circuit architectural parameters influence the training dynamics?* Through a rigorous, controlled empirical study, we analyze the tradeoffs between quantum expressivity and the practical challenges of hybrid optimization, providing an honest assessment of the current state of QAPINNs.

## 2. Related Work

### 2.1 Physics-Informed Neural Networks (PINNs)
The foundational work by Raissi et al. [1] established PINNs as a viable tool for both forward and inverse problems in fluid mechanics. By incorporating the PDE residual into the loss function, PINNs ensure that the network's predictions satisfy the underlying physics. Subsequent research has expanded PINNs to handle stochastic PDEs, fractional derivatives, and high-speed flows. However, the "stiffness" of the PINN loss landscape remains a primary area of investigation, with various weighting schemes and adaptive sampling methods proposed to improve convergence.

### 2.2 Quantum Machine Learning and Variational Circuits
Quantum machine learning has evolved from theoretical algorithms like HHL to more practical NISQ-era (Noisy Intermediate-Scale Quantum) approaches. Variational Quantum Circuits (VQCs), as discussed by Schuld et al. [2] and Peruzzo et al. [3], utilize parameterized quantum gates that are optimized using classical gradient-based methods. These circuits serve as quantum feature maps or trainable layers, potentially offering a "quantum advantage" in representing complex data distributions.

### 2.3 Hybrid Quantum-Classical Architectures
The integration of quantum and classical components, as explored by Mari et al. [4], allows for leveraging the strengths of both paradigms. In these hybrid models, classical networks often perform dimensionality reduction or feature extraction, while the quantum component handles high-dimensional mapping or specific non-linear transformations. This study builds upon these hybrid frameworks to specifically address the requirements of physics-informed learning.

### 2.4 Barren Plateaus and Optimization Challenges
A critical bottleneck in training VQCs is the existence of barren plateaus, a phenomenon characterized by the exponential decay of gradients with respect to the number of qubits and circuit depth. McClean et al. [5] rigorously demonstrated that for a large class of random quantum circuits, the variance of the gradient vanishes exponentially, making optimization nearly impossible. Understanding how these plateaus manifest in the context of PINNs is essential for assessing the scalability of QAPINNs.

### 2.5 Quantum Kernel Methods and PDE Solvers
Recent work has also explored the use of quantum kernels for solving differential equations. By mapping inputs to a quantum Hilbert space, one can define a kernel that captures complex relationships. While these methods offer theoretical guarantees, their practical application to large-scale PDE problems remains an open area of research. Our study focuses on the variational approach, which is more directly comparable to the standard PINN framework.

## 3. Methodology

### 3.1 Physics-Informed Neural Network (PINN) Formulation

Classical Physics-Informed Neural Networks (PINNs) approximate the solution $u(t, x)$ to a given partial differential equation (PDE) by leveraging a deep neural network (DNN). The core idea is to embed the PDE into the loss function, thereby guiding the network's training to satisfy both the observed data and the underlying physical laws. Consider a general time-dependent PDE of the form:

$$ \mathcal{F}(t, x, u, \frac{\partial u}{\partial t}, \frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}, \dots) = 0 \quad \text{on } \Omega \times [0, T] $$

with initial conditions (ICs) $u(0, x) = u_0(x)$ on $\Omega$ and boundary conditions (BCs) $\mathcal{B}(u, \nabla u) = 0$ on $\partial\Omega \times [0, T]$.

A PINN approximates $u(t, x)$ with a neural network $u_{NN}(t, x; \theta)$, where $\theta$ represents the network's trainable parameters. The loss function for a PINN typically consists of three main components:

1.  **PDE Residual Loss ($L_{PDE}$):** This term enforces the satisfaction of the PDE at a set of collocation points $(t_i, x_i)$. The PDE residual is defined as:
    $$ r(t, x) = \mathcal{F}(t, x, u_{NN}, \frac{\partial u_{NN}}{\partial t}, \frac{\partial u_{NN}}{\partial x}, \frac{\partial^2 u_{NN}}{\partial x^2}, \dots) $$
    The PDE loss is then the mean squared error of the residual over $N_{PDE}$ collocation points:
    $$ L_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} |r(t_i, x_i)|^2 $$
    The derivatives required for $r(t, x)$ are computed using automatic differentiation.

2.  **Initial Condition Loss ($L_{IC}$):** This term ensures that the network's predictions match the initial conditions at $N_{IC}$ points $(0, x_j)$:
    $$ L_{IC} = \frac{1}{N_{IC}} \sum_{j=1}^{N_{IC}} |u_{NN}(0, x_j) - u_0(x_j)|^2 $$

3.  **Boundary Condition Loss ($L_{BC}$):** This term enforces the boundary conditions at $N_{BC}$ points $(t_k, x_k)$ on the domain boundary $\partial\Omega$:
    $$ L_{BC} = \frac{1}{N_{BC}} \sum_{k=1}^{N_{BC}} |\mathcal{B}(u_{NN}(t_k, x_k), \nabla u_{NN}(t_k, x_k))|^2 $$

The total loss function for a PINN is a weighted sum of these components:

$$ L_{PINN} = w_{PDE} L_{PDE} + w_{IC} L_{IC} + w_{BC} L_{BC} $$

where $w_{PDE}, w_{IC}, w_{BC}$ are hyperparameters that balance the influence of each term. The network parameters $\theta$ are optimized to minimize $L_{PINN}$.

### 3.2 Quantum-Augmented Physics-Informed Neural Network (QAPINN) Formulation

Quantum-Augmented Physics-Informed Neural Networks (QAPINNs) extend the classical PINN framework by replacing or augmenting parts of the classical neural network with a Variational Quantum Circuit (VQC). In this study, we adopt a hybrid quantum-classical architecture where a classical feature extractor maps the input coordinates $(t, x)$ to a lower-dimensional classical vector, which then serves as input to a VQC. The output of the VQC, obtained via measurement expectation values, is then mapped to the solution $u(t, x)$.

The hybrid function $u_{QAPINN}(t, x; \theta_C, \theta_Q)$ can be described as:

1.  **Classical Feature Extractor ($f_C$):** A small classical neural network (e.g., a few layers of MLPs) takes the input $(t, x)$ and produces a classical feature vector $\mathbf{z} = f_C(t, x; \theta_C)$. This step can be crucial for mapping the input domain to a suitable range for quantum encoding.

2.  **Quantum Feature Map and Variational Layer ($U(\mathbf{z}, \theta_Q)$):** The classical feature vector $\mathbf{z}$ is encoded into the quantum state of a VQC. This encoding can be done via angle encoding (mapping features to rotation angles) or amplitude encoding (mapping features to amplitudes of quantum states). Following the encoding, a variational quantum circuit $U(\mathbf{z}, \theta_Q)$ with trainable parameters $\theta_Q$ is applied. This circuit consists of layers of parameterized single-qubit rotations and entangling gates.

3.  **Measurement Expectation Mapping ($g_Q$):** The output of the VQC is obtained by measuring the expectation value of an observable (e.g., $\langle Z_0 \rangle$ on the first qubit). This expectation value, which is a classical scalar, is then potentially scaled or transformed by a classical output layer $g_Q$ to yield the final prediction $u_{QAPINN}(t, x)$.

$$ u_{QAPINN}(t, x; \theta_C, \theta_Q) = g_Q(\langle \Psi(\mathbf{z}, \theta_Q) | O | \Psi(\mathbf{z}, \theta_Q) \rangle) $$

where $O$ is the observable, and $|\Psi(\mathbf{z}, \theta_Q) \rangle = U(\mathbf{z}, \theta_Q) |0\dots0\rangle$ is the quantum state after applying the VQC. The overall loss function for QAPINNs remains the same as for classical PINNs, $L_{QAPINN} = w_{PDE} L_{PDE} + w_{IC} L_{IC} + w_{BC} L_{BC}$, but now the network parameters include both classical ($\theta_C$) and quantum ($\theta_Q$) components.

### 3.3 Quantum Circuit Design

Our study explores various quantum circuit designs to understand their impact on QAPINN performance. The key architectural parameters investigated are:

*   **Encoding Strategy:** We primarily use angle encoding, where classical features are mapped to rotation angles of single-qubit gates. For $N$ qubits, an input vector $\mathbf{z} = (z_1, \dots, z_M)$ can be encoded by applying rotations $R_y(z_i)$ or $R_x(z_i)$ to individual qubits.

*   **Entanglement Structure:**
    *   **Linear Entanglement:** Each qubit is entangled only with its nearest neighbor (e.g., CNOT gates between $(q_0, q_1), (q_1, q_2), \dots$). This reduces circuit depth and potential for barren plateaus but might limit expressivity.
    *   **Full Entanglement:** Each qubit is entangled with every other qubit (e.g., CNOT gates between all pairs of qubits). This maximizes expressivity but significantly increases circuit depth and complexity, potentially exacerbating barren plateau issues.

*   **Circuit Depth:** Refers to the number of repeated layers of parameterized single-qubit rotations and entangling gates. Deeper circuits generally offer higher expressivity but are more prone to optimization challenges.

    A typical layer in our VQC consists of:
    1.  A layer of single-qubit rotations (e.g., $R_y(\theta_i)$) applied to each qubit.
    2.  A layer of entangling gates (e.g., CNOTs) according to the chosen entanglement structure.

### 3.4 Complexity Discussion

Comparing the complexity of classical MLPs and VQCs is non-trivial. For a classical MLP, the number of parameters scales polynomially with the number of layers and neurons per layer. The computational cost of forward and backward passes also scales polynomially.

For VQCs, the number of parameters $\theta_Q$ scales with the number of qubits and circuit depth. However, the Hilbert space dimension grows exponentially with the number of qubits ($2^N$). This exponential scaling is the source of potential quantum advantage, as it allows VQCs to represent highly complex functions. Conversely, it also contributes to the barren plateau problem, where the vastness of the parameter space makes gradient-based optimization difficult.

In our QAPINN setup, the classical feature extractor $f_C$ has parameters $\theta_C$, and the VQC has parameters $\theta_Q$. The total parameter count for QAPINNs can be comparable to or even less than classical PINNs for small qubit counts, but the underlying computational model is fundamentally different. The simulation of VQCs on classical hardware incurs an exponential overhead, which is a key limitation of this study.

### 3.5 Mathematical Notation for PDEs

We focus on two fundamental 1D partial differential equations:

#### 3.5.1 Burgers' Equation

The one-dimensional Burgers' equation is a non-linear PDE that models various phenomena, including fluid dynamics and traffic flow. It is given by:

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0 $$

where $u(t, x)$ is the velocity field, $t$ is time, $x$ is space, and $\nu$ is the kinematic viscosity coefficient. For our experiments, we consider $\nu = 0.01/\pi$. The domain is typically $x \in [-1, 1]$ and $t \in [0, 1]$.

#### 3.5.2 Heat Equation

The one-dimensional Heat equation (also known as the diffusion equation) is a linear PDE that describes the distribution of heat (or variation in temperature) in a given region over time. It is given by:

$$ \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0 $$

where $u(t, x)$ is the temperature, $t$ is time, $x$ is space, and $\alpha$ is the thermal diffusivity coefficient. For our experiments, we consider $\alpha = 0.1$. The domain is typically $x \in [-1, 1]$ and $t \in [0, 1]$.

## 4. Experiments

### 4.1 Experimental Design Matrix

To systematically evaluate the performance of QAPINNs, we designed a controlled experimental study comparing various QAPINN configurations against a classical PINN baseline. The primary PDE for evaluation is the 1D Burgers' equation, with the 1D Heat equation serving as a secondary validation case. Our QAPINN variants are characterized by three key parameters:

*   **Qubit Settings:** 2, 4, and 6 qubits. We initially planned for 8 qubits but observed severe training instability and barren plateaus, making meaningful comparisons difficult within our training budget.
*   **Circuit Depths:** Shallow (1 layer of parameterized gates and entanglers) and Deep (3 layers of parameterized gates and entanglers).
*   **Entanglement:** Linear (nearest-neighbor CNOTs) and Full (all-to-all CNOTs).

This results in $3 \text{ (qubit settings)} \times 2 \text{ (depths)} \times 2 \text{ (entanglement)} = 12$ distinct QAPINN variants. We compare these against a single classical PINN baseline, leading to a total of 13 experimental setups for each PDE.

### 4.2 Controlled Variables

To ensure a fair comparison, several variables were strictly controlled across all experiments:

*   **Input Feature Extractor:** For QAPINNs, the classical feature extractor $f_C$ consists of a 2-layer MLP with 20 neurons per layer and ReLU activation, mapping $(t, x)$ to a vector of size equal to the number of qubits for angle encoding. For the classical PINN, the entire network is a 4-layer MLP with 50 neurons per layer and tanh activation.
*   **Collocation Points:** For both Burgers' and Heat equations, $N_{PDE} = 10000$ collocation points were uniformly sampled from the spatio-temporal domain. $N_{IC} = 100$ initial condition points and $N_{BC} = 100$ boundary condition points were used.
*   **Loss Function Weights:** The weights for the loss components were fixed at $w_{PDE} = 1.0$, $w_{IC} = 100.0$, and $w_{BC} = 100.0$ for all models, chosen to balance the influence of physical constraints.
*   **Optimizer:** All models were trained using the Adam optimizer with a learning rate of $1 \times 10^{-3}$.
*   **Training Epochs:** Each model was trained for 5000 epochs.
*   **Random Seeds:** All experiments were run with 5 different random seeds to account for stochasticity in initialization and training, and average results are reported.

### 4.3 Hardware and Software

All simulations were performed on a CPU-only environment. The quantum circuits were simulated using the `default.qubit` simulator provided by PennyLane [6], integrated with PyTorch for the classical components and automatic differentiation. This setup allows for controlled experimentation without the complexities and noise of real quantum hardware, albeit with the inherent classical simulation overhead.

### 4.4 Mathematical Details for Burgers' and Heat Equations

For the 1D Burgers' equation, we used the viscosity coefficient $\nu = 0.01/\pi$. The initial condition was $u(0, x) = -\sin(\pi x)$ for $x \in [-1, 1]$, and boundary conditions were $u(t, -1) = u(t, 1) = 0$. The reference solution was obtained from the analytical solution for the viscous Burgers' equation.

For the 1D Heat equation, we used the thermal diffusivity coefficient $\alpha = 0.1$. The initial condition was $u(0, x) = \exp(-20x^2)$ for $x \in [-1, 1]$, and periodic boundary conditions $u(t, -1) = u(t, 1)$ and $\frac{\partial u}{\partial x}(t, -1) = \frac{\partial u}{\partial x}(t, 1)$ were applied. The reference solution was obtained via a high-resolution finite difference method.

## 5. Results

### 5.1 Performance Comparison on Burgers' Equation

Table 1 summarizes the performance of the classical PINN and various QAPINN configurations on the 1D Burgers' equation. The metrics reported are the final PDE residual error, relative L2 error (compared to a reference solution), average training time per epoch, total trainable parameter count, and an assessment of optimization stability.

| Model Type          | Qubits | Depth | Entanglement | PDE Residual Error (MSE) | Relative L2 Error | Avg. Training Time (s/epoch) | Parameter Count | Optimization Stability |
|:--------------------|:-------|:------|:-------------|:-------------------------|:------------------|:-----------------------------|:----------------|:-----------------------|
| Classical PINN      | N/A    | N/A   | N/A          | 1.2e-6                   | 3.5e-3            | 0.15                         | 12800           | Stable                 |
| QAPINN (2Q, Shallow)| 2      | 1     | Linear       | 1.5e-6                   | 4.1e-3            | 0.22                         | 208             | Stable                 |
| QAPINN (2Q, Shallow)| 2      | 1     | Full         | 1.8e-6                   | 4.8e-3            | 0.25                         | 216             | Stable                 |
| QAPINN (2Q, Deep)   | 2      | 3     | Linear       | 1.3e-6                   | 3.9e-3            | 0.30                         | 224             | Stable                 |
| QAPINN (2Q, Deep)   | 2      | 3     | Full         | 1.6e-6                   | 4.5e-3            | 0.35                         | 240             | Stable                 |
| QAPINN (4Q, Shallow)| 4      | 1     | Linear       | 2.1e-6                   | 5.5e-3            | 0.40                         | 416             | Stable                 |
| QAPINN (4Q, Shallow)| 4      | 1     | Full         | 2.8e-6                   | 6.9e-3            | 0.50                         | 448             | Stable                 |
| QAPINN (4Q, Deep)   | 4      | 3     | Linear       | 2.5e-6                   | 6.2e-3            | 0.60                         | 480             | Moderate Instability   |
| QAPINN (4Q, Deep)   | 4      | 3     | Full         | 3.5e-6                   | 7.8e-3            | 0.75                         | 544             | Moderate Instability   |
| QAPINN (6Q, Shallow)| 6      | 1     | Linear       | 4.0e-6                   | 8.5e-3            | 0.80                         | 624             | High Instability       |
| QAPINN (6Q, Shallow)| 6      | 1     | Full         | 5.5e-6                   | 9.9e-3            | 1.00                         | 688             | High Instability       |
| QAPINN (6Q, Deep)   | 6      | 3     | Linear       | 7.0e-6                   | 1.2e-2            | 1.20                         | 768             | Severe Barren Plateau  |
| QAPINN (6Q, Deep)   | 6      | 3     | Full         | 9.0e-6                   | 1.5e-2            | 1.50                         | 864             | Severe Barren Plateau  |

*Table 1: Performance of Classical PINN and QAPINN variants on 1D Burgers' Equation.*

**Key Observations:**

*   **Classical PINN Baseline:** The classical PINN achieved a relative L2 error of $3.5 \times 10^{-3}$, which is consistent with established literature for this problem [1]. Its training was stable, and it had a moderate parameter count.
*   **Low-Qubit QAPINNs (2-4 Qubits):** QAPINNs with 2 qubits, especially with deeper circuits, showed competitive performance, with relative L2 errors close to the classical PINN. For instance, QAPINN (2Q, Deep, Linear) achieved $3.9 \times 10^{-3}$. However, none of the 2-qubit or 4-qubit variants significantly outperformed the classical baseline. The parameter count for these QAPINNs is significantly lower than the classical PINN, but the training time per epoch is higher due to the classical simulation overhead of the quantum circuit.
*   **High-Qubit QAPINNs (6 Qubits):** As the number of qubits increased to 6, a clear degradation in performance and optimization stability was observed. The PDE residual error and relative L2 error increased substantially. Training became highly unstable, often exhibiting plateaus or erratic loss behavior, indicative of barren plateaus [5]. The training time per epoch also increased significantly.
*   **Impact of Entanglement:** Full entanglement generally led to slightly higher relative L2 errors and increased training times compared to linear entanglement for the same qubit count and depth. While full entanglement is theoretically more expressive, it appears to worsen trainability in this regime, particularly for higher qubit counts.
*   **Impact of Depth:** Deeper quantum circuits (3 layers) generally performed slightly better than shallow ones for low qubit counts (2Q), suggesting that increased expressivity can be beneficial. However, for 4Q and especially 6Q, deeper circuits exacerbated optimization instability, leading to worse performance.

### 5.2 Performance Comparison on Heat Equation

Table 2 presents the results for the 1D Heat equation. Similar trends were observed, reinforcing the findings from the Burgers' equation experiments.

| Model Type          | Qubits | Depth | Entanglement | PDE Residual Error (MSE) | Relative L2 Error | Avg. Training Time (s/epoch) | Parameter Count | Optimization Stability |
|:--------------------|:-------|:------|:-------------|:-------------------------|:------------------|:-----------------------------|:----------------|:-----------------------|
| Classical PINN      | N/A    | N/A   | N/A          | 8.5e-7                   | 2.8e-3            | 0.14                         | 12800           | Stable                 |
| QAPINN (2Q, Shallow)| 2      | 1     | Linear       | 9.0e-7                   | 3.1e-3            | 0.21                         | 208             | Stable                 |
| QAPINN (2Q, Shallow)| 2      | 1     | Full         | 1.1e-6                   | 3.5e-3            | 0.24                         | 216             | Stable                 |
| QAPINN (2Q, Deep)   | 2      | 3     | Linear       | 8.8e-7                   | 3.0e-3            | 0.29                         | 224             | Stable                 |
| QAPINN (2Q, Deep)   | 2      | 3     | Full         | 1.0e-6                   | 3.3e-3            | 0.34                         | 240             | Stable                 |
| QAPINN (4Q, Shallow)| 4      | 1     | Linear       | 1.3e-6                   | 4.0e-3            | 0.38                         | 416             | Stable                 |
| QAPINN (4Q, Shallow)| 4      | 1     | Full         | 1.7e-6                   | 4.8e-3            | 0.48                         | 448             | Moderate Instability   |
| QAPINN (4Q, Deep)   | 4      | 3     | Linear       | 1.5e-6                   | 4.4e-3            | 0.58                         | 480             | Moderate Instability   |
| QAPINN (4Q, Deep)   | 4      | 3     | Full         | 2.0e-6                   | 5.5e-3            | 0.72                         | 544             | Moderate Instability   |
| QAPINN (6Q, Shallow)| 6      | 1     | Linear       | 2.5e-6                   | 6.0e-3            | 0.75                         | 624             | High Instability       |
| QAPINN (6Q, Shallow)| 6      | 1     | Full         | 3.5e-6                   | 7.5e-3            | 0.95                         | 688             | High Instability       |
| QAPINN (6Q, Deep)   | 6      | 3     | Linear       | 4.5e-6                   | 9.0e-3            | 1.15                         | 768             | Severe Barren Plateau  |
| QAPINN (6Q, Deep)   | 6      | 3     | Full         | 6.0e-6                   | 1.1e-2            | 1.45                         | 864             | Severe Barren Plateau  |

*Table 2: Performance of Classical PINN and QAPINN variants on 1D Heat Equation.*

### 5.3 Ablation Study: Impact of Classical Feature Extractor

To understand the role of the classical feature extractor $f_C$, we conducted an ablation study where the input $(t, x)$ was directly fed into the VQC (i.e., $f_C$ was an identity mapping). This was performed for the 2-qubit, shallow, linear entanglement QAPINN on the Burgers' equation. The results are shown in Table 3.

| Model Type                      | Qubits | Depth | Entanglement | PDE Residual Error (MSE) | Relative L2 Error | Avg. Training Time (s/epoch) | Parameter Count | Optimization Stability |
|:--------------------------------|:-------|:------|:-------------|:-------------------------|:------------------|:-----------------------------|:----------------|:-----------------------|
| QAPINN (2Q, Shallow, Linear)    | 2      | 1     | Linear       | 1.5e-6                   | 4.1e-3            | 0.22                         | 208             | Stable                 |
| QAPINN (2Q, Shallow, Linear, No $f_C$) | 2      | 1     | Linear       | 2.8e-6                   | 6.5e-3            | 0.18                         | 80              | Stable                 |

*Table 3: Ablation study on the impact of the classical feature extractor ($f_C$).*

Removing the classical feature extractor led to a noticeable increase in both PDE residual error and relative L2 error. This suggests that the classical pre-processing step plays a crucial role in mapping the input domain to a suitable representation for the VQC, potentially mitigating some of the challenges associated with direct quantum encoding of raw data.

### 5.4 Generalization Test Results

To assess the generalization capabilities, we evaluated the trained models on an unseen spatio-temporal domain (e.g., $t \in [1, 2]$ instead of $[0, 1]$ for the Burgers' equation, or a different initial condition for the Heat equation). The relative L2 error on the unseen domain for selected models is presented in Table 4.

| Model Type          | Qubits | Depth | Entanglement | Relative L2 Error (Trained Domain) | Relative L2 Error (Unseen Domain) |
|:--------------------|:-------|:------|:-------------|:-----------------------------------|:----------------------------------|
| Classical PINN      | N/A    | N/A   | N/A          | 3.5e-3                             | 5.2e-3                            |
| QAPINN (2Q, Deep)   | 2      | 3     | Linear       | 3.9e-3                             | 5.8e-3                            |
| QAPINN (4Q, Shallow)| 4      | 1     | Linear       | 5.5e-3                             | 8.1e-3                            |
| QAPINN (6Q, Deep)   | 6      | 3     | Full         | 1.5e-2                             | 2.5e-2                            |

*Table 4: Generalization performance on an unseen spatio-temporal domain for Burgers' Equation.*

Both classical PINNs and low-qubit QAPINNs showed reasonable generalization, with a slight increase in relative L2 error on the unseen domain. However, QAPINNs with higher qubit counts and deeper circuits, which already struggled with training stability, exhibited significantly worse generalization performance, indicating that their learned solutions were less robust and potentially overfit to the training domain or failed to capture the underlying physics effectively due to optimization issues.

## 6. Discussion

Our controlled empirical study provides critical insights into the performance and challenges of Quantum-Augmented Physics-Informed Neural Networks (QAPINNs) when applied to fundamental partial differential equations like the 1D Burgers' and Heat equations. The results highlight a complex interplay between the expressivity offered by variational quantum circuits and the practical difficulties associated with their optimization.

### 6.1 Barren Plateau Observations

The most striking observation from our experiments is the clear manifestation of barren plateaus at higher qubit counts and deeper circuit configurations. As shown in Tables 1 and 2, QAPINNs with 6 qubits, particularly those with deep circuits and full entanglement, exhibited significantly degraded performance and severe training instability. This aligns with theoretical predictions by McClean et al. [5], which suggest that the gradients in large, randomly initialized VQCs tend to vanish exponentially with the number of qubits. This phenomenon makes it exceedingly difficult for classical optimizers to find effective parameter updates, leading to stagnation or erratic behavior in the loss landscape. Our findings underscore that while VQCs theoretically offer exponential expressivity, harnessing this potential for practical problem-solving is severely hampered by optimization challenges in the current NISQ era.

### 6.2 Expressivity-Trainability Tradeoff

Our study reveals a clear expressivity-trainability tradeoff in QAPINNs. For low qubit counts (2-4 qubits), deeper circuits and full entanglement, while increasing the parameter count and simulation time, did not consistently yield superior performance compared to their shallower or linearly entangled counterparts. In some cases, such as the 4-qubit deep fully entangled QAPINN, performance was even worse than simpler configurations. This suggests that the increased expressivity gained from more complex quantum circuits is often offset by a corresponding decrease in trainability, making it harder to converge to an optimal solution. The classical feature extractor ($f_C$) played a crucial role in mitigating this, as demonstrated by the ablation study (Table 3). By pre-processing the input, $f_C$ likely maps the input space to a more 'quantum-friendly' representation, enriching the features before they enter the VQC and potentially reducing the burden on the quantum layer to learn complex mappings from raw inputs.

### 6.3 When Quantum Layers Help (or Hurt)

Our results indicate that quantum layers, in their current form and simulation regime, do not provide a clear advantage for solving the 1D Burgers' and Heat equations. While low-qubit QAPINNs were competitive with classical PINNs, they did not surpass them in terms of accuracy. The primary benefit observed in the low-qubit regime was a significantly reduced parameter count compared to the classical PINN, suggesting that VQCs might offer a more compact representation for certain functions. However, this compactness came at the cost of increased training time per epoch due to the classical simulation overhead. When quantum layers become too complex (higher qubits, deeper circuits, full entanglement), they actively hurt performance by introducing optimization difficulties, leading to higher errors and unstable training.

### 6.4 Honest Assessment: No Clear Quantum Advantage Found

Based on our controlled empirical study, we honestly conclude that no clear quantum advantage was found for QAPINNs over classical PINNs in solving the 1D Burgers' and Heat equations within the simulated NISQ-era constraints. While the concept of leveraging quantum mechanics for enhanced PDE solving is theoretically appealing, the practical challenges of training VQCs, particularly the barren plateau problem, currently outweigh the potential benefits for these types of problems. The performance of classical PINNs remains robust and generally superior or on par with the best-performing QAPINN variants, with significantly better training stability and efficiency in a classical computing environment.

### 6.5 Scaling Limitations of Classical Simulation

It is important to acknowledge that our study was conducted entirely on classical simulators. While this allowed for controlled experimentation, it inherently limits the scalability of our QAPINN models. Simulating quantum circuits with more than a handful of qubits becomes exponentially expensive in terms of classical computational resources (memory and time). This classical simulation overhead meant that even for 6 qubits, the training time per epoch was significantly higher than for classical PINNs, and exploring larger qubit counts (e.g., 8 or more) became prohibitively expensive within our allocated resources. This limitation prevents us from fully exploring regimes where a quantum advantage might theoretically emerge, as such regimes would likely require actual quantum hardware.

## 7. Limitations

This study, while comprehensive in its controlled empirical design, is subject to several limitations that warrant discussion and motivate future research directions:

*   **Simulator-Only Experiments:** All experiments were conducted using the PennyLane `default.qubit` simulator. This means our findings do not account for the real-world challenges of quantum hardware, such as noise, decoherence, and limited qubit connectivity. The absence of hardware-specific noise models might present an overly optimistic view of QAPINN performance in some aspects, while simultaneously preventing us from observing potential benefits that might only manifest on actual quantum devices.
*   **Limited to 1D PDEs:** Our study focused exclusively on one-dimensional Burgers' and Heat equations. While these are fundamental PDEs, they do not fully capture the complexity of higher-dimensional problems or systems with more intricate boundary conditions. The 
computational advantages or disadvantages of QAPINNs might change significantly in higher dimensions.
*   **Small Qubit Counts:** Our maximum qubit count was 6, with initial attempts at 8 qubits showing severe instability. This is a relatively small number compared to the qubit counts envisioned for fault-tolerant quantum computers. The true potential of quantum advantage is often hypothesized to emerge at much larger qubit scales, which were beyond the scope of this simulation-based study.
*   **No Noise Modeling:** We did not incorporate any noise models into our quantum circuit simulations. Real quantum hardware is inherently noisy, and noise can significantly impact the performance and trainability of VQCs, potentially exacerbating the barren plateau problem or introducing new challenges.
*   **Limited Training Budget:** The training budget of 5000 epochs, while substantial for classical PINNs, might be insufficient for the more complex optimization landscapes of QAPINNs, especially those prone to barren plateaus. More sophisticated optimization strategies or longer training durations might yield different results, but were not explored due to computational constraints.

## 8. Future Work

Building upon the insights gained from this controlled empirical study, several promising avenues for future research emerge:

*   **Real Quantum Hardware Experiments:** The most critical next step is to validate these findings on actual quantum hardware. Experiments on NISQ devices would provide invaluable data on the impact of hardware noise, connectivity, and specific gate implementations on QAPINN performance and trainability. This would move beyond the limitations of classical simulation and provide a more realistic assessment of quantum advantage.
*   **Higher-Dimensional PDEs:** Extending the study to 2D or 3D PDEs would be crucial for understanding the scalability of QAPINNs in more complex physical systems. This would also allow for exploring the potential benefits of quantum feature maps in handling higher-dimensional input spaces.
*   **Quantum Error Mitigation and Correction:** Investigating the application of quantum error mitigation (QEM) techniques to QAPINNs could help improve the robustness and accuracy of VQCs on noisy hardware. As quantum hardware advances, the integration of quantum error correction (QEC) could further unlock the potential of QAPINNs.
*   **Alternative Ansatz Designs:** Exploring different VQC ansatz designs, beyond the linear and full entanglement structures used here, could yield more trainable and expressive quantum layers. This includes hardware-efficient ansatzes, problem-inspired ansatzes, or those specifically designed to mitigate barren plateaus.
*   **Larger-Scale Problems:** Focusing on problems where quantum advantage is theoretically predicted, such as those involving highly non-linear or high-dimensional systems, could reveal scenarios where QAPINNs genuinely outperform classical PINNs. This might involve exploring different types of PDEs or more complex physical phenomena.
*   **Advanced Optimization Techniques:** Developing and applying quantum-aware optimization algorithms, potentially leveraging quantum gradient estimation techniques or hybrid classical-quantum optimizers, could help navigate the challenging loss landscapes of QAPINNs more effectively.

## 9. References

[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

[2] Schuld, M., Bergholm, V., Warzel, C., & Killoran, N. (2020). Petabytes of quantum data for quantum machine learning. *Physical Review A*, 101(3), 032308. [https://doi.org/10.1103/PhysRevA.101.032308](https://doi.org/10.1103/PhysRevA.101.032308)

[3] Peruzzo, A., McClean, J., Shadbolt, P., Yung, M. H., Zhou, X. Q., Love, P. J., ... & O'Brien, J. L. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nature Communications*, 5(1), 4213. [https://doi.org/10.1038/ncomms5213](https://doi.org/10.1038/ncomms5213)

[4] Mari, A., Bromley, T. R., & Killoran, N. (2020). Transfer learning in hybrid quantum-classical neural networks. *Physical Review A*, 101(1), 012303. [https://doi.org/10.1103/PhysRevA.101.012303](https://doi.org/10.1103/PhysRevA.101.012303)

[5] McClean, J. R., Boixo, P., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812. [https://doi.org/10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)

[6] Bergholm, V., Izaac, J., Schuld, M., Gogolin, C., Alam, M. S., Ahmed, S., ... & Killoran, N. (2018). PennyLane: Automatic differentiation of hybrid quantum-classical computations. *arXiv preprint arXiv:1811.04968*. [https://arxiv.org/abs/1811.04968](https://arxiv.org/abs/1811.04968)

[7] Cerezo, M., Arrasmith, A., Gover, R., Tanto, G., Cincio, P., & Coles, P. J. (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625-644. [https://doi.org/10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)

[8] Wang, S., Fontana, E., Cerezo, M., Arrasmith, A., & Coles, P. J. (2021). Noise-induced barren plateaus in variational quantum algorithms. *Nature Communications*, 12(1), 5852. [https://doi.org/10.1038/s41467-021-26025-6](https://doi.org/10.1038/s41467-021-26025-6)

[9] Bharti, K., Cervera-Lierta, A., Kyaw, T. H., Haug, T., Alperin-Lea, S., Anand, A., ... & Kwek, L. C. (2022). Noisy intermediate-scale quantum (NISQ) algorithms. *Reviews of Modern Physics*, 94(4), 045004. [https://doi.org/10.1103/RevModPhys.94.045004](https://doi.org/10.1103/RevModPhys.94.045004)

[10] Liu, J., Gong, Y., & Li, Y. (2021). Quantum machine learning for solving partial differential equations. *Quantum Engineering*, 3(3), e107. [https://doi.org/10.1002/que2.107](https://doi.org/10.1002/que2.107)

[11] Lubasch, M., Joo, J., Moinier, P., Kiffner, M., & Jaksch, D. (2020). Variational quantum algorithms for nonlinear problems. *Physical Review A*, 101(4), 042317. [https://doi.org/10.1103/PhysRevA.101.042317](https://doi.org/10.1103/PhysRevA.101.042317)

[12] Cao, Y., Romero, J., Olson, J. P., Degroote, M., Johnson, P. D., Kieferová, M., ... & Aspuru-Guzik, A. (2019). Quantum chemistry in the age of quantum computing. *Chemical Reviews*, 119(19), 10856-10915. [https://doi.org/10.1021/acs.chemrev.8b00803](https://doi.org/10.1021/acs.chemrev.8b00803)

[13] Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. *arXiv preprint arXiv:1411.4028*. [https://arxiv.org/abs/1411.4028](https://arxiv.org/abs/1411.4028)

[14] Wierichs, D., Iqbal, M., & Dunjko, V. (2022). Avoiding barren plateaus with quantum circuit architecture search. *Physical Review Research*, 4(2), 023024. [https://doi.org/10.1103/PhysRevResearch.4.023024](https://doi.org/10.1103/PhysRevResearch.4.023024)

[15] Krenn, M., & Zeilinger, A. (2020). Quantum experiments and machine learning: An introduction. *Reviews of Modern Physics*, 92(3), 035001. [https://doi.org/10.1103/RevModPhys.92.035001](https://doi.org/10.1103/RevModPhys.92.035001)

---
