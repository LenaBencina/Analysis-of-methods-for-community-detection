# Project description

This is the practical part of my master's thesis research where the main subject is the comparison of methods for community detection. The focus was on methods adjusted for directed and weighted networks which made the analysis even more complex. The main reference is the paper [Clustering and Community Detection in Directed Networks: A Survey](https://arxiv.org/abs/1308.0971).

**Abstract**

*In this work we present an important part of network analysis, the problem of community
detection. Hidden community structure, which we aim to reconstruct using
community detection methods, contains important information about the underlying
graph structure. The main focus of this work is to analyze the consideration
of edge direction, which is usually ignored because of its complicated nature. We
offer an exhaustive review of the problem and corresponding methods on intuitive
as well as on formal level.*

*An additional difficulty we face is the unclear definition of the problem. We
explore various views of the problem definition with the detailed analysis including
the presentation of the main approaches dealing with the problem. Additionally, we
focus on four different methods, each dealing with directed and weighted networks
on its own way. Methods we include are the well-known Louvain method, Leiden,
Infomap and OSLOM. An important part of the work is the empirical comparative
analysis of the presented methods based on a number of detected communities,
accuracy, stability and value of modularity in synthetic as well as in real networks.*


# About the code

We used four different methods (Louvain, Leiden, Infomap and OSLOM) on each of the three networks; two real ([Email network](https://snap.stanford.edu/data/email-Eu-core.html) and [Retweet network](http://dirros.openscience.si/IzpisGradiva.php?id=922&lang=eng&prip=rul:1469976:d1)) and a set of synthetic [LFR networks](https://arxiv.org/abs/0805.4770).

Each of the files *CD-LFR-network.py*, *CD-email-network.py*, *CD-retweet-network.py* includes the comparison of methods for different input network described above.

All of the main functions used in the analysis are in the file *CD-functions.py*.

The file *Network-visualizations.py* is used for generating network visualizations.


----------------------------------------------------------------------------------------------------------------------------

If you want to use my code and need help, do not hesitate to write me.


