# Resnet

ResNet over Convolutional Networks and different Normalization schemes
Residual Networks (ResNet) [2] present a very simple idea to introduce identity
mappings via residual connections. They are shown to significantly improve
the quality of training (and generalization) in deeper networks. We covered the
core idea in class. Before starting this part of the assignment, you should thoroughly read the ResNet paper. Specifically, we will implement the ResNet [2]
architecture, and study the effect of different normalisation schemes, viz. Batch
Normalization [3], Instance Normalization [7], Batch-Instance Normalization
[7], Layer Normalization [1], and Group Normalization [8] within ResNet. We
will experiment with a dataset on Indian Birds species classification.
1.1 Image Classification using Residual Network
This sub-part will implement ResNet for Image Classification.
Dataset: link
1. You will implement a ResNet architecture from scratch in PyTorch. Assume that the total number of layers in the network is given by 6n+2.
This includes the first hidden (convolution) layer processing the input of
size 256×256. This is followed by n layers with feature map size 256×256,
followed by n layers with feature map size 128×128, n layers with feature
map size given by 64 × 64, and finally a fully connected output layer with
r units, r being number of classes. The number of filters in the 3 sets of
n hidden layers (after the first convoluational layer) are 16, 32, 64, respectively. There are residual connections between each block of 2 layers,
except for the first convolutional layer and the output layer. All the convolutions use a filter size of 3 × 3 inspired by the VGG net architecture [6].
1
Whenever down-sampling, we use the convolutional layer with stride of 2.
Appropriate zero padding is done at each layer so that there is no change
in size due to boundary effects. The final hidden layer does a mean pool
over all the features before feeding into the output layer. Refer to Section
4.2 in the ResNet paper for more details of the architecture. Your program
should take n as input. It should also take r as input denoting the total
number of classes.
2. Train a ResNet architecture with n = 2 as described above on the given
dataset. Use a batch size of 32 and train for 50 epochs. For dataset, r
= 25. Use SGD optimizer with initial learning rate of 10−4
. Decay or
schedule the learning rate as appropriate. Feel free to experiment with
different optimizers other than SGD.
3. Train data has 30,000 images while validation has 7500 images each image
is of different resolution. Report the following statistics / analysis:
• Accuracy, Micro F1, Macro F1 on Train, Val and Test splits.
• Plot the error, accuracy and F1 (both macro and micro) curves for
both train and val data
2 Impact of Normalization
The standard implementation of ResNet uses Batch Normalization [3]. In this
part of the assignment, we will replace Batch Normalization with various other
normalization schemes and study their impact.
1. Implement from scratch the following normalization schemes. They should
be implemented as a sub-class of torch.nn.Module.
(a) Batch Normalization (BN) [3]
(b) Instance Normalization (IN) [7]
(c) Batch-Instance Normalization (BIN) [4]
(d) Layer Normalization (LN) [1]
(e) Group Normalization (GN) [8]
2. In your implementation of ResNet in Section 1.1, replace the Pytorch’s
inbuilt Batch Normalization (nn.BatchNorm2d) with the 5 normalization
schemes that you implemented above, giving you 5 new variants of the
model. Note that normalization is done immediately after the convolution
layer. For comparison, remove all normalization from the architecture,
giving you a No Normalization (NN) variant as a baseline to compare
with. In total, we have 6 new variants (BN, IN, BIN, LN, GN, and NN).
3. Train the 6 new variants on the dataset, as done in Section 1.1
2
4. As a sanity check, compare the error curves and performance statistics of
the model trained in Section 1.1 with the BN variant trained in this part.
It should be identical (almost).
5. Compare the error curves and performance statistics (accuracy, micro F1,
macro F1 on train / val / test splits) of all six models.
6. Impact of Batch Size: [8] claim that one of the advantages of GN over
BN is its insensitivity to batch size. Retrain the BN and GN variants of the
model with Batch Size 8 and compare them with the same variants trained
with Batch Size 128. Note that reducing the batch size will significantly
increase the training time. To reduce the time taken, run it for 50 epochs
and early stop based on validation accuracy.
7. Visualize model’s thinking: An important part of any model training
is analysing why a model predicts any particular label for a given input.
In this sub-part we’ll use Grad-CAM [5] for producing visual-explanations
from our ResNet models. Grad-CAM computes the gradient of input feature map with respect to the specified output class. Higher the gradient
value at particular position indicates it’s importance for predicting particular class. For more information please read Grad-CAM paper[5]. For
this assignment use the Grad-CAM’s implementation from this package.
• Consider following 7 classes for visualization: Cattle Egret , Coppersmith Barbet, Indian Peacock, Red Wattled Lapwing, Ruddy Shelduck, White Breasted Kingfisher, White Breasted Waterhen. Use
the best performing model from all the variants trained above, visualize the gradient maps (superimposed on image) of 5 correctly
and 5 incorrectly classified images for each of the above class from
the validation set. Gradient needs to be computed with respect to
the correct class and for the output feature map from last 2n layers
(where feature map size is 64x64). These gradient maps need to be
mean aggregated to give final 64x64 gradient map. The given package does all of these by default and you just need to mention the
model layers.
• Analyse the gradient maps and report your observations.
