# Predicting the Accuracy of a Few-Shot Classifier

This repository contains the code of the experiments performed in the following paper

[Predicting the Accuracy of a Few-Shot Classifier]()

by Myriam Bontonou, Louis Bethune and Vincent Gripon

## Abstract
In the context of few-shot learning, one cannot evaluate generalization using validation sets, due to the small amount of training examples. In this paper, we are interested in finding alternatives to answer the question: is my classifier generalizing well to previously unseen data? We first analyze the reasons for the variability of generalization performances. We then investigate the case of using transfer-based solutions, and consider three settings: i) supervised where we only have access to a few labeled samples, ii) semi-supervised where we have access to both a few labeled samples and a set of unlabeled samples and iii) unsupervised where we only have access to unlabeled samples. For each setting, we propose reasonable measures that we empirically demonstrate to be correlated with the generalization ability of considered classifiers. We also show that these simple measures can be used to predict generalization up to a certain confidence. We conduct our experiments on standard few-shot vision datasets.

## Usage
### 1. Dependencies
- Python >= 3.6
- Pytorch >= 1.2

### 2. Download Datasets

The datasets should be download at TODO

The "features" folder should be placed at the root of the project.  

### 3. Download Pretrained Models
Our code is built upon the trained models of [SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning](https://arxiv.org/pdf/1911.04623.pdf) and [Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://openaccess.thecvf.com/content_WACV_2020/papers/Mangla_Charting_the_Right_Manifold_Manifold_Mixup_for_Few-shot_Learning_WACV_2020_paper.pdf).

### 4. Evaluate the correlation between metrics and the generalization performance
#### a. Supervised setting
#### b. Semi-supervised setting
#### c. Unsupervised setting

### 5. Predict the generalization performance from metrics
#### a. Supervised setting
#### b. Semi-supervised setting
#### c. Unsupervised setting

### 6. Case study on mini-ImageNet
#### Dependencies
Numerous dependencies are required, in addition to Python and Pytorch:
```
pygsp >= 0.5.1
python-louvain >= 0.14
networkx >= 2.3
scipy >= 1.3.1
scikit-learn >= 0.21
```

#### Generate the graphs
The code to generate the graph on densenet-m backbone with novel split is:
```
python3 main_smooth.py --mode=monitoring_volume --dataset=densenet-m-novel --n_way=2
```

The dot files will be produced and store in `graphs` folder.  
To compute the correlation between the edge weights and the accuracy of logistic regression:
```
python3 main_smooth.py --mode=monitoring_volume --dataset=densenet-m-base --dot-name=graphs/louvain_dendrogram_communities_1_20.dot --n_way=5 --n_shot=5 --n_val=595
```

For the corresponding bipartite graph:
```
python3 main_smooth.py --mode=monitoring_volume --dataset=densenet-m-base\&densenet-m-novel --n_way=2
```

To compute corelation between the edges of two graphs:
```
python3 dot_correlation.py --graph_name_1=pathdot1 --graph_name_2=pathdot2
```

## Contact
Please contact us if there are any problems.

Myriam Bontonou (myriam.bontonou@imt-atlantique.fr)
