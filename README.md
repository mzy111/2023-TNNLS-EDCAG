# EDCAG

Using the code, please cite:
Wang J, Ma Z, Nie F, et al. Efficient Discrete Clustering With Anchor Graph[J]. IEEE Transactions on Neural Networks and Learning Systems, early access, June 08, 2023, doi: 10.1109/TNNLS.2023.3279380.

Paper URL: https://ieeexplore.ieee.org/abstract/document/10146387

The code explanation: 
The main function of the code: EDCAG.m
You can use EDCAG_test.m to perform EDCAG clustering for PenDigits data set.
If you have any questions, please connect zhenyu.ma@mail.nwpu.edu.cn

# Use of Main Function

Anchor Generation: [B,Anchor] = ULGEmzy(X,log2(m),k,selAnchor) 

n: the number of instances in primal data; d: the number of dimensions in primal data;

X: primal data matrix; m: the number of anchors; k: the number of nearest neighbors; selAnchor: the way to select anchors 1 -- BKHK 2 -- K-means++ 3 -- K-means 4 -- Random Selection;

B: anchor graph (n \times m); Anchor: anchor data matrix (m \times d); 

Discrete Clustering: [labelnew,~] = EDCAG(B,c);

c: the number of real classes; labelnew: predicted labels;

example on PenDigits data set with 512 anchors (BKHK generation) and 50 neighbors:

load('PenDigits_data.mat') % load Data
c = length(unique(label));
[B,Anchor]=ULGEmzy(X,log2(512),50,1); % Anchor Generation
[labelnew,~] = EDCAG(B,c); % Discrete Clustering
result = ClusteringMeasure_All(label,labelnew); % Evaluation
