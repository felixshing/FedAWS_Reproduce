# Reimplementation of FedAWS 
This is a reimplementation of FedAWS (Federated Learning with Only Positive Labels, ICML 2020) in PyTorch. 
The reimplementation algorithm comes from [FedRepo](https://github.com/lxcnju/FedRepo).
Basically, I just modified the code under feddata.py to make sure each client holds only positive label, 
and I also implement the loss function implemented in [FedFace](https://github.com/illidanlab/fedface/blob/main/source/FedAwS.py), which also uses the FedAWS algorithm.

However, the performance of this reimplementation is pretty bad. It can only achieve at most 17% accuracy on cifar-10 dataset.
I try to change the hyperparameters, but it doesn't work. I think the reason is that the this code doesn't implement the
`the learning rate multiplier of the spreadout loss λ` in the FedAWS algorithm, making the  class embeddings are not spreadout.
But I don't know how to implement it. If you have any idea, please let me know. Thanks.
You can reach me by email: rcheng4@gmu.edu. I would love to discuss this algorithm and anything related to FL with you.


## Running Tips
Just use `python train_lab.py` then you can run FedAWS under cifar-10 dataset.
You can change hyperparameters in `train_lab.py`.



## Personal Homepage
  * [Homepage](https://felixshing.github.io/)



## Datasets
Only cifar-10 is on this repo. But you can also use cifar-100 and FaMNIST datasets after you download them.

## Acknowledgement
This repo is based on [FedRepo](https://github.com/lxcnju/FedRepo). Thanks for the author's great work.
