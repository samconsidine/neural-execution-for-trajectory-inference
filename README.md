# neural-execution-for-trajectory-inference
Neural algorithmic reasoning for pseudotime trajectory inference

Model:
----
- [X] Autoencoder
- [X] Neuralised clustering
- [X] Experiment with IDEC models
- [X] Neural execution engine for Prim's algo
- [ ] Full MST Loss functions
    - [X] Reconstruction loss
    - [ ] Structure-preserving loss
- [X] Solve MST loss bias
- [X] Figure out regularisation balance
- [X] Check gradients as they pass back through all components of the system
- [X] Save and load trained models, different functionality for both
- [X] Write tests for neural execution
- [X] Find alternative to cdist for calculating distance between pairs of points
- [X] Make models GPU ready
- [X] Figure out line segment projection of cells
- [X] Figure out cross-dimensions for MST recon loss
- [X] Train models
- [X] Integrate synthetic datasets
- [X] Implement neuarlised KMeans in ACE style

Benchmarking:
------------
- [X] Benchmarking functionality. Compare to other models (or at least VITAE)
- [X] Neeed benchmarking suite that can make some table of performances of models
- [X] Plots demonstrating performance
- [X] Find issues with implementation
- [X] Case sutudies of strengths/weaknesses
- [X] Overall benchmarking study

Benchmarking outputs
-------------------
- [X] Table of results, my model vs others 
- [X] Plot examples of model working
- [X] ...