---
layout: post
title:  "Parallel state-vector emulator of quantum circuits. Pure Rust implementation."
date:   2022-06-26 20:22:50 +0300
categories: Numerical-and-quantum
usemathjax: true
---
This blogpost is dedicated to a simple Rust based implementation of a quantum circuits emulator with several nice performance features that I will gradually introduce throughout the text. The first question that arises is why do we need to use Rust as a core language? Why don't we use python frameworks such Jax or Julia language that are equipped with easy parallelization, jit compilation, and designed for efficient scientific computing? Let us discuss this question in details.

Why Rust?
=========

Rust is a system programming language that includes several strong features at once. It maintains zero-cost abstraction principle and minimal runtime without garbage collecting as C/C++ and also has the same performance as C/C++. Meanwhile it is almost impossible to get undefined behaviour at runtime until you do not use unsafe blocks in your code. It is also thread safe which means that you could implement all sorts of parallelism and concurrency without being afraid of data races. The list of Rust advantages is quite long and we are not going to discuss all of them in here. The most important for us are those that are already listed above.

Now let us discuss why these features allow one to implement a better state-vector emulator in comparison with Python/Julia based ones. First of all, a state-vector emulator core is essentially a matrix-matrix multiplication module with some specific features 1) typically it is a multiplication of a small matrix with a very long and narrow one. In case of two-qubit gate it is  a multiplication of a matrix of size $$4 \times 4$$ (gate matrix) with a matrix of size $$4\times 2^{(n-2)}$$ (state), where $$n$$ is the number of qubits; 2) It requires to use non-trivial slicing of a state, because you may apply a gate to an arbitrary pair of state indices. It is not enough to maintain a row major and a column major matrix layouts or even layouts with arbitrary strides. Thus, one can take advantage from the purely custom implementation of this matrix-matrix multiplication. For example, it would be profitable to have the following features in our custom implementation: 1) keep only one heap allocated buffer for the state vector and update it in-place; 2) keep all gates on a stack; 3) organize parallel and efficient traversal of the state vector without changing its memory layout. All these require low-level control of memory that is provided by Rust. Meanwhile, the fearless parallelism together with easy to use and fast iterators provided by Rust are also very helpful. As one can see, Rust serves as a nice candidate to be the core language for such kind of things. Now, let us dive into the implementation.

Quantum state memory layout
===========================

...............................