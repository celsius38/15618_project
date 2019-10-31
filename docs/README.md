# Parallel Density-based Clustering Algorithm
### Sailun Xu (sailunx)
### Yueni Liu (yuenil)

at: [github](https://github.com/celsius38/15618_project)

## Summary

We are going to implement an optimized density-based clustering algorithm (DBSCAN: Density-based spatial clustering of applications with noise [1]) on both GPU (CUDA) and distributed memory (MPI), and perform a detailed analysis and comparison on various datasets.

## Background

DBSCAN is a density-based clustering algorithm. Each object is labelled as core, border or noise as shown in the graph below. We use Euclidean distance to define proximity. The algorithm has two parameters, namely R which is the proximity radius, and MinPts which is the minimum number of neighbors. If a given objects has more neighbors than MinPts within distance R, then it is defined as core and all objects reachable directly or indirectly from it is labelled as border and they are within the same cluster. If the object is not reachable from any core, then it is classified as noise.


<div style="text-align:center"><img src ="image/dbscan.png" /></div>

<div style="text-align:center"><img src ="image/dbscan_pseudo.png" /></div>

There are many efforts put on parallelizing DBSCAN, like G-DBSCAN [2], PDS-DBSCAN [3], MR-DBSCAN [4], NG-DBSCAN [5], RP-DBSCAN [6], etc. They are targeting for different platforms like GPU, OpenMP, MPI, MapReduce etc. In this project, we will focus on graph-based implementation with GPU and MPI.
<span style="color:red"> What aspects of the problem might be benefit from parallelism and why?
</span>

## Challenge
<span style="color:red"> 
What do you hope to learn by doing the project?   
What aspects of the problem might make it difficult to parallelize?   
What are the dependencies?   
What are its memory access characteristics?    (is there locality? is there a high communication to computation ratio?)   
Is there divergent execution?   
Platform constraints?   

</span>

## Resources
We will use both GHC machines and Lateday clusters.

We will use G-DBSCAN [2] as our starting point and implement the algorithm from scratch before optimizing it.

For the purpose of performance analysis, we will use the datasets (100, 000 points each) applied by the original DBSCAN paper as shown in the graph below. 

<div style="text-align:center"><img src ="image/cluster_type.png" /></div>

We will also use real-life datasets such as OpenStreetMap (GPS data), Cosmo50 (N-body simulation data), and TeraClickLog (click log data).

## Goals and Deliverables
Must achieve: optimize existing parallel DBSCAN with CUDA on single GHC machine (expected speedup by 50x over sequential version, and 5x over other parallel implementation).

Plan to achieve: reduce communication cost and adapted the algorithm to distributed memory machines with MPI.

Hope to achieve: experiments on different datasets especially extremely large one that can not be fit in memory.

What to demonstrate: this project is more research oriented instead of application oriented, so there won’t be an interactive demo. We are able to show the clustering results on real-life datasets such as OpenStreetMap and Cosmo50. We’ll also show performance analysis graphs including speedup, time breakdown, load balance, scalability to the number of thread/nodes, and memory footprint. 

<span style="color:red"> What are you hoping to learn about the workload or system being studied?</span>

## Platform choice
The algorithm G-DBSCAN we chose as a starting point is targeting for GPU, so it’s better if we do optimization on the same platform for performance comparison. However, the size of dataset has been growing rapidly in real life, so that a single machine can not meet the requirement in most cases. So it’s meaningful and necessary to be able to efficiently run clustering algorithm on distributed machines. That’s why we also plan to adapt the algorithm for message passing model running on a cluster of machines. 


## Schedule 
| Week                | Goal                               | Detail                                                               |
|---------------------|------------------------------------|----------------------------------------------------------------------|
| Week 1(10/29-11/05) | Research                           | Write proposal, read related paper and implement sequential version. |
| Week 2(11/05-11/12) | 1st Parallel Implementation        | Implement G-DBSCAN with CUDA and do analysis.                        |
| Week 3(11/12-11/19) | Optimization (Checkpoint!)         | Conduct optimization and write checkpoint report.                    |
| Week 4(11/19-11/26) | MPI Version                        | Improve and implement MPI version running on cluster.                |
| Week 5(11/26-12/03) | Performance analysis               | Run experiments on different datasets and draw graphs.               |
| Week 6(12/03-12/10) | Wrap up (Final report and poster!) | Run more experiments and prepare final report and poster.            |
## References

[1] Ester, M., Kriegel, H.P., Sander, J. and Xu, X., 1996, August. A density-based algorithm for discovering clusters in large spatial databases with noise. In Kdd (Vol. 96, No. 34, pp. 226-231).

[2] Andrade, G., Ramos, G., Madeira, D., Sachetto, R., Ferreira, R. and Rocha, L., 2013. G-dbscan: A gpu accelerated algorithm for density-based clustering. Procedia Computer Science, 18, pp.369-378.

[3] Patwary, M.A., Palsetia, D., Agrawal, A., Liao, W.K., Manne, F. and Choudhary, A., 2012, November. A new scalable parallel DBSCAN algorithm using the disjoint-set data structure. In Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (p. 62). IEEE Computer Society Press.

[4] He, Y., Tan, H., Luo, W., Mao, H., Ma, D., Feng, S. and Fan, J., 2011, December. Mr-dbscan: an efficient parallel density-based clustering algorithm using mapreduce. In 2011 IEEE 17th International Conference on Parallel and Distributed Systems (pp. 473-480). IEEE.

[5] Lulli, A., Dell'Amico, M., Michiardi, P. and Ricci, L., 2016. NG-DBSCAN: scalable density-based clustering for arbitrary data. Proceedings of the VLDB Endowment, 10(3), pp.157-168.

[5]Song, H. and Lee, J.G., 2018, May. RP-DBSCAN: A superfast parallel DBSCAN algorithm based on random partitioning. In Proceedings of the 2018 International Conference on Management of Data (pp. 1173-1187). ACM.
