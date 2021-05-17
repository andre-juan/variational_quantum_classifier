# Variational Quantum Classifier

In this repository, I implement a **Variational Quantum Classifier (VQC)**, trained on the [iris dataset.](https://archive.ics.uci.edu/ml/datasets/iris) The algorithm is implemented from scratch, using only basic tools from Qiskit, with the goal of actually understanding how the model works, and what it's doing.

A VQC is also called a **Parameterized Quantum Circuit**, which refers to the way these models work: they are quantum circuits whose gates (most often, rotations) are parameterized by free parameters in a parameter vector. These parameters are determined via the minimization of a loss function, calculated on data - that is, the parameters are *learned through data*. Hence, this is an example of a **quantum machine learning** algorithm. However, the optimization of the loss function is carried out classically, which is why we often refer to VQCs as **hybrid (classical-quantum) machine learning models**.

Schematically, a VQC circuit can be summarized in 3 components, or layers:

- **Feature map**: part of the circuit responsible for encoding classical data into quantum states which will be processed in the quantum circuit by the quantum algorithm;
- **Variational layer**: parameterized part of the circuit. Parameters in this layer are those learned in the training process;
- **Measurement**: final part of the circuit, consisting of measurements of the quantum register(s), thus producing classical information.

The feature map and variational layers can be constructed in several different ways (that is, different ansätze). The choice of particular constructions is often refered to as the **architecture** of the VQC.

As aforementioned, VQC is a hybrid algorithm, because it uses classical routines to optimize the parameters according to the training data, via the construction of a classical loss/cost function, directly from the measurements of the quantum registers in the last layers of the circuit (which produces classical data).

VQC is often also refered to as a **Quantum Neural Network**, because of the resemblance between some aspects of its structure and training procedure with respect to that of classical neural networks, which are also layered models whose parameters are determined via the minimization of a loss function evaluated on data.

If we draw this analogy, one could say that VQC is a neural network whose forward propagation is quantum (performed by a quantum circuit), whilst its backpropagation (actual optimization step for the minimization of the loss) is performed classically. In context, this analogy is enough for one to call VQC a "quantum neural network". But, personally, I think this denomination (implying this analogy) may cause some confusion, for the following reason: one major ingredient of classical neural networks (in great part responsible for its outstanding performance) is the non-linearity introduced by actiavtion functions on the units.

Now, all gates performed in a quantum circuit are **linear, unitary operators**. Although there are some proposals of implementing non-liniearities in quantum circuits (see, for example [this paper](https://arxiv.org/abs/1806.06871), or [this one](https://arxiv.org/abs/1808.10047) which deals with photonic quantum computing), which would indeed reproduce all major ingredients of a classical neural network (included non-linear activation functions) to be implemented in a quantum computer, this is not a trivial task to be implemented on general architectures. And this is not what we will implement in this notebook. Thus, I prefer to avoid the "quantum neural network" denomination, as I think it can be a bit misleading, specitally if taken out of context.

For more details on VQC, and a more formal and illustrated version of the description above, please see [this notebook!]().

In this repository, we build simple implementations of VQC with the main goal of understanding what the model is actually doing. Thus, iven its simplicity, its performance probably won't be that great, and that's indeed not the primary goal here. Fot better performing models, I suggest [Qiskit's](https://qiskit.org/documentation/tutorials/machine_learning/03_vqc.html) or [PennyLane's](https://pennylane.ai/qml/demos/tutorial_variational_classifier.html) implementations of variational quantum classifiers. In these implementations, you find not only more flexibility on the model architecture, but also different (and potentially better) ways to perform the optimization (learning) procedure.

The code in this repository was inspired by [this video](https://youtu.be/5Kr31IFwJiI), check it out!
______________

This repository contains the following files:

- `vqc_iris_fundamentals.ipynb`: Notebook with a step-by-step implementation of the VQC algorithm, as well as markdown cells containing detailed explanations of each step. The algorithm is trained in the iris dataset.
- `vqc.py`: Python file with all functions of the implementation of VQC. This file contains generalized versions of the functions constructed in the notebook above, which allows more flexibility in the experimentation of different architectures and hyperparameters for the model construction. This is done in the notebook below.
- `vqc_iris_experiments.ipynb`: Notebook with a grid-search like experimentation of different model architectures and hyperparameters, using the functions defined in the python file above.
