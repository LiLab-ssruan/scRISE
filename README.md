# scRISE
Our work introduces a novel clustering strategy for scRNA-seq data called scRISE, which uses an autoencoder to iteratively denoise the data (with Laplacian smoothing) and construct the cell-graph reliably, while in the meantime seamlessly incorporating cell graph information with a self-supervised discriminative embedding technique that allows identifying correct clusters through adaptively determined similarity threshold.
A distinctive feature of scRISE is its use of an iterative cycle smoothing approach to achieve optimal clustering results during the data reconstruction phase. Through the application of a self-supervised discriminative embedding learning technique, scRISE guides the clustering of the reconstructed data, ensuring a more precise and insightful representation of the underlying cellular structures. 

You can use the following command to run, and the sample data has been placed in the data folder.

python train.py --data Deng --run 3 --seed 1111
