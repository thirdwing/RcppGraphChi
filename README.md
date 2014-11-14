RcppGraphChi: R interface to GraphChi
============

**This package is in very beginning stage (Just one SVD function). Much more work should be done before release.**

GraphChi is a spin-off of the [GraphLab]( http://www.graphlab.org )-project from the Carnegie Mellon University.
It is based on research by [Aapo Kyrola]( http://www.cs.cmu.edu/~akyrola/) and his advisors. 

GraphChi can run very large graph computations on just a single machine,
by using a novel algorithm for processing the graph from disk (SSD or hard drive). 

In some cases GraphChi can solve bigger problems in reasonable time than many other available *distributed* frameworks.
GraphChi also runs efficiently on servers with plenty of memory, and can use multiple disks in parallel by striping the data.

GraphChi is implemented in plain C++, and available as open-source under the flexible Apache License 2.0.

You can find the source code from the [official repo](https://github.com/GraphChi/graphchi-cpp).

GraphChi provides a wonderful opportunity to solve the memory bottleneck in R, which reads all data into memory by default.

RcppGraphChi is based on Rcpp and RcppEigen.

Testing datasets download link: http://www.select.cs.cmu.edu/code/graphlab/datasets/
