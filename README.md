# Verilog Implementation of MLB (XNOR-Popcount-Based Dot Product)

This repository contains Verilog modules developed to simulate and understand the hardware architecture presented in the paper's **Multi-Level Binary (MLB)** accelerator.

## Overview

The design explores the computation of binary dot products using:

* XNOR operations
* Popcount units
* Ripple Carry Adders (RCA)
* Accumulators
* Reduction (Tree Adder) networks

The goal is to model the dataflow of the MLB architecture and study how binary neural network computations can be efficiently implemented in digital hardware.

## Repository Structure

* `full_adder.v` – 1-bit full adder
* `ripple_carry_adder.v` – Parameterized N-bit ripple carry adder
* `xnor_popcount_4_bit.v` – 4-bit XNOR-Popcount implementation
* `testbenches/` – Simulation testbenches
* Additional modules for reduction trees, accumulators, and scaling units will be added as development progresses.

## Features

* Parameterized Verilog modules
* Synthesizable RTL
* Hierarchical hardware design
* Modular and reusable building blocks
* Suitable for experimentation with binary neural network accelerators

## Tools

The code has been tested using standard Verilog simulation environments such as:

* Icarus Verilog
* GTKWave
* ModelSim / QuestaSim

## Objective

This repository serves as a learning and experimentation platform for understanding:

* Binary Neural Network (BNN) hardware
* XNOR-Popcount computation
* Adder architectures
* Reduction trees
* Digital VLSI design concepts

## References

The implementation is inspired by the MLB architecture described in the corresponding research paper and is intended for educational and research purposes.
