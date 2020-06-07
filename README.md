# RSHN
The implementation of our AAAI 2020 paper "[GSSNN: Graph Smoothing Splines Neural Network](https://www.researchgate.net/publication/337548602_GSSNN_Graph_Smoothing_Splines_Neural_Networks)". [Slides](http://ddl.escience.cn/f/VbTm).

# Requirements
python == 3.6.2<br>
torch == 1.1.0<br>
numpy == 1.16.4<br>
scipy == 1.2.0<br>
networkx == 2.2<br>
torch_scatter == 1.3.0<br>
torch_geometric == 1.3.0

# How to use
  ### Dataset
  The data folder includes our propocessed data for training and testing. <br>
  The orginal datasets can be founded from [here](https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets).

  ### Model
  The model folder includes our proposed model "GSSNN".<br>
  The model/utils folder includes graph utils and Scaled Smoothing Splines module used in model.<br>
  The model/process_data file processes data and computes the graph centrality.<br>
  The torch_geometeric/nn/pool folder includes the designed NodeImportance layer used in model.<br>
  The torch_geometeric/nn/conv folder includes the convolutional layers used in model provides by [torch_geometeric](https://github.com/rusty1s/pytorch_geometric) library.<br>
  
  ### Prameters Setting
  dim: the hidden dimension of node feature<br>
  conv_layer: the number of convolutional layer<br>
  ss_layer: the number of smoothing splines layer<br>
  Mi: the number of knot used in smoothing splines layer i<br>
  epsilon: used in smoothing splines to guarantee the denominator non-zero<br>
  add_knot: whether to consider the important nodes features as residual connection to the graph-level representation<br>
  
  ### Training/Testing
  ```
  cd model
  python process_data.py --dataset MUTAG
  python GSSNN.py --dataset MUTAG --batch_size 128 --lr 0.01 --weight_decay 5e-4 --dim 32 --conv_layer 3 --ss_layer 2 --M1 5 --M2 5 --epsilon 1e-6 --add_knot True --epoch 100
  python GSSNN_10_folds.py --dataset MUTAG --batch_size 128 --lr 0.01 --weight_decay 5e-4 --dim 32 --conv_layer 3 --ss_layer 2 --M1 5 --M2 5 --epsilon 1e-6 --add_knot True --epoch 100
  ```
  
  
# Citation
```
@inproceedings{zhu2020GSSNN
author={Shichao Zhu and Lewei Zhou and Shirui Pan and Chuan Zhou and Guiying Yan and Bin Wang},
title={GSSNN: Graph Smoothing Splines Neural Network},
journal={Proceedings of the 34th AAAI Conference on Artificial Intelligence},
year={2020}
}
```
