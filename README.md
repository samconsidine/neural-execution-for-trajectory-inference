# neural-reasoning-trajectory-inference
Neural algorithmic reasoning for pseudotime trajectory inference

TODO:
----
- [X] Save and load trained models
- [X] Find bug in neural execution
- [X] Write tests for neural execution
- [/] Find alternative to cdist for calculating distance between pairs of points
- [ ] Integrate synthetic datasets
- [ ] Implement neuarlised KMeans 

Bugs:
-----
- [X] MST solver is doing very little
- [X] Centroids initialised between 0 and 0.5
- [ ] Scale inputs to neural exec solver to be from same distribution as target data (to work accross AE training procedure and dataset)