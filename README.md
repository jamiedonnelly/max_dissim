# max_dissim

**Maximum Dissimilarity data selection**

Python algorithm to iteratively add datapoints from a dataset $B$ to a subset $A$ such that each new data point $i$ maximises the euclidean distance between the point $i$ and the points currently in $A$.

<img src="https://github.com/jamiedonnelly/max_dissim/blob/main/MDA.gif" align="center"/>

Beginning with a single datapoint, new points ($Z$) are iteratively added (moving from $B$ to $A$) such that each new point $Z$ maximises the distance between the current points in $A$. 

The aim is to capture as much of the behaviour of a dataset in the fewest number of points and can be used as an alternative to some other experimental design such as Latin Hypercube Sampling. 
