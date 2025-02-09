# fourier_learning_ibm

## What's this?
This is a code to reproduce the results of arxiv:hogehoge.

## How to use
### 1. Determine parameters

Initially, we determine two parameters, specifically Trotter steps $n_{\text{step}}$ and the maximum evolution time $t_K$. This process is carried out using the [get_param.ipynb](get_param.ipynb) notebook.

### 2. Create dataset

The next step is to create a training dataset using the [get_dataset.ipynb](get_dataset.ipynb) notebook. A dataset is a set of input Hamiltonians, $H_i$, and output observed values, $y(H_i)$.

### 3. Calculate features

After that, we calculate a real part and imaginary part of $\bra{\psi}e^{-iHt_k}\ket{\psi}$, where $k=0, \cdots, K$, $\ket{\psi}$ is a state where the middle half is 1, and the rest is 0. In the case of 8 qubits for example, $\ket{\psi} = \ket{00111100}$. To do so, we use
- [fourier_feature_exact.ipynb](fourier_feature_exact.ipynb)
    - calculate features numerically (without Trotterization)
- [fourier_feature_sim_noiseless.ipynb](fourier_feature_sim_noiseless.ipynb)
    - calculate features using Trotterization with a noise-free classical simulator (AerSimulator)
- [fourier_feature_sim_noisy.ipynb](fourier_feature_sim_noisy.ipynb)
    - calculate features using Trotterization with a noisy classical simulator (AerSimulator)
- [fourier_feature_qpu.ipynb](fourier_feature_qpu.ipynb)
    - calculate features using Trotterization with IBM's QPU.

### 4. Perform regression

Finally, we run a regression to fit the features to the training data, and display the result, using [regression.ipynb](regression.ipynb).


