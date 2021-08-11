# Neural Network IR0 Generator

For the series of vgg neural network (vgg11, vgg13, vgg16, vgg19), I think they devide convolutions in 5 sections in which the number of channels are the same. And pooling layers are located after each section. The difference is the number of convolutions in each section. See `main.cpp` and `model.cpp` to see how to construct a vgg-like neural network (without dense layer). 

To generate the input, please refer to `run.sh`.

Note: Needs large memory and storage. 