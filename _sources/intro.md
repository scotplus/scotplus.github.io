# Welcome to SCOOTR

## Description

SCOOTR is an optimal transport based tool for aligning datasets that share an underlying manifold. While many other algorithms of this type seek to learn the underlying manifold and use it as a common embedding space to indirectly align each dataset, SCOOTR learns sample and feature correspondences directly while implicity learning this manifold structure. 

We have found SCOOTR to be particularly relevant to single-cell multi-omics datasets. Gathering multiple sets of omics data on the same sample set of cells can be quite expensive (co-assays) or impossible. Rather than enabling these expensive experimental methods, SCOOTR seeks to solve the problem computationally by allowing biologists to align separately assayed data as if they were co-assayed. Considering datasets coming from the same cell population often share a common embedding space or underlying manifold, SCOOTR solves this problem with state-of-the-art accuracy. While we focus on single-cell multi-omics datasets in these tutorials, we believe that SCOOTR may be applicable to more problems of a similar variety. Reach out to us if SCOOTR has helped you successfully align some other type of unpaired data!

The following tutorials contain enough information to help you understand optimal transport for the purpose of aligning your data, including its few hyperparameters and optimization procedure. In addition, they will help you understand how our package works in practice on real world datasets. Equipped with the knowledge of how to use our package theoretically and in practice, you will be able to apply SCOOTR to any datasets you may have.

We would like to thank the Executable Book Project for making this draft possible.

## Table of Contents

```{tableofcontents}
```
