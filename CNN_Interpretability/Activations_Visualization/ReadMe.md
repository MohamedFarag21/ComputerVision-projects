# Visualizing CNNs' activations are one of the most important and intuitive techniques to understand what are the outputs from your layers.
***First of all, thanks to Francois Chollet for his great book and explanations, part of the code at this repo was developed by him at his book***
<code><img src="https://i.pinimg.com/originals/7b/db/5e/7bdb5e02f5b896af975897a4b5adc4f2.png" width="256"  height="256"></code>

## 1- What are the activations ?
Activations are the output from the CNN layers, also they are called the output feature maps, which mainly generated after using the convolutional operation.

### SO here we develop the steps easily to get out outputs:
1- First choose your model.
2- load your data.
3- Choose the layers you want to visualize its activations.
4- Generate the activations maps.
5- Visualize (**The part of the code built by Francois ❤**)

# Results
At this trial, i utilized ResNet50V2, pretrained on ImageNet weights to predict the activations for the data i used for my research.

# Observations as made by Francois Chollet.

1- The first conv layers near the inputs are having information regarding the whole scene as we can see:
**This is the first Convolutional Layers**

<code><img src="https://github.com/MohamedFarag21/ComputerVision-projects/blob/main/CNN_Interpretability/Activations_Visualization/Images/conv1_conv.png?raw=true" width="933"  height="264"></code>

Less abstract features, but we will see with increasing the depth the activations become more abstract and less interpretable, while being too sparse.

**We go deeper at the network...not all activations are on, that means those filters couldn't find the pattern!**

<code><img src="https://github.com/MohamedFarag21/ComputerVision-projects/blob/main/CNN_Interpretability/Activations_Visualization/Images/conv2_block1_1_relu.png?raw=true" width="933"  height="264"></code>

*As Francois explaining: "A deep neural
network effectively acts as an information distillation pipeline, with raw data going in
(in this case, RGB pictures) and being repeatedly transformed so that irrelevant information
is filtered out (for example, the specific visual appearance of the image), and
useful information is magnified and refined (for example, the class of the image)."*

**Deeper at the network...**

<code><img src="https://github.com/MohamedFarag21/ComputerVision-projects/blob/main/CNN_Interpretability/Activations_Visualization/Images/conv3_block1_3_conv.png?raw=true" width="939"  height="1786"></code>

**At this image we are the third resnent block at the network, we lost the scene and the model focused more on the features related to the class.**

Hope you enjoy it! ❤️









