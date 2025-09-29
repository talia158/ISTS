# Visual Influence Spread Test Suite

An application for testing new and existing Influence Spread heuristics 
on large network graphs (1,000,000+ nodes) and visualizing the spread of 
influence across a network. 

---

## Features
- Support for user-inputted heuristics written in Python3, user-inputted graphs and psuedo-random graph generation (Watts-Strogatz and Barabasi Albert)
- Node and edge aggregation, done with Louvain clustering ran on RAPIDS cuGraph
(requires supported Nvidia GPU)
- Cluster heatmaps (tinted by fraction of infected nodes in a cluster)
- Edge highlighting (edges which propagate influence at a timestep glow in cyan)
- Recursive declustering of nodes (the two features above hold when a node is expanded)

---

![demo](https://github.com/talia158/ISTS/blob/main/demo.gif?raw=true)
