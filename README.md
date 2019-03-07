~~ <script type="text/x-mathjax-config">
~~ MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});
~~ </script>
~~ <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
# Cross-edge Computation Offloading
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/NarcissusHliangZhao/Cross-edge-Computation-Offloading/blob/master/LICENSE.txt)

A python package of Cross-edge Computation Offloading (CCO) algorithm and its distributed version, Decor for Mobile Edge Computing (MEC). 

This package is an implementation of the proposed offloading algorithm for mobility-aware computation-intensive partitionable applications. 
Specifically, for a non-convex edge site-selection sub-problem, we propose a Sampling-and-Classification-based (SAC) algorithm to obtain the 
near optimal solution. Based on Lyapunov optimization CCO algorithm is proposed to jointly determine edge site-selection and energy harvesting 
without a priori knowledge. The transmission, execution and coordination cost, as well as the penalty for task failure, are chosen as performace 
metrics. 

**Citation**:
> [The corresponding paper is under review now. Leave this empty temporarily.]

**Used dataset**:
> EUA dataset @ https://github.com/swinedge/eua-dataset/.

**Referenced Code**:
> RACOS @ https://github.com/eyounx/RACOS.


## Quick Start
### Code Structure
+ python package **racos**: 
This package contains the algorithm named Racos, which is a specific algorithm based on Sampling-And-Classification (SAC) Framework. Details can be 
found at http://lamda.nju.edu.cn/yuy/research_sal.ashx.
***

+ directory **dataset**:
This directory contains the real-life dataset of base stations in Melbourne CBD area.
***

+ python package **cross_edge_offloading**:
This package contains the algorithms proposed in our work. The package named **cco** is the implementation of main algorithm. The package named 
**benchmarks** contains three benchmark policies for comparsion.

### Results
![Simulation Results #1](./figures/cost.png)
![Simulation Results #2](./figures/battery.png)
