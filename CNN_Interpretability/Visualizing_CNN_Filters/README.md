# Visualizing Convnet filters

*The aim is to understand what is the pattern at the image that a CNN filter responds to ?*
**Thanks to Francois chollet for providing a way to understand deeper and simply without complications, the code used at this part was developed by him, at his book "Deep Lerarning with python 2nd edition."**

<code><img src="https://images.manning.com/360/480/resize/book/a/2a49d38-96e5-4bf7-8555-57f689c52ebf/Chollet-2ed-HI.png" width="256"  height="256"></code>

# How can we do it ?

1. Generate a random image.
2. Pass the image to the model.
3. Get the filter you want to understand the pattern it encodes.
4. Get the output.
5. Measure if the filter response is totaly maximized.
6. Utilize _Gradient_ascent_ to maximize the response of the filter.
7. Back propagtion is used to modify the pixel values to find "**The image that will maximize the filter's response**"

# Results
Here we can see the results of some filter pattern at the `block3_conv1` at the VGG16 network


<code><img src="https://github.com/MohamedFarag21/ComputerVision-projects/blob/main/CNN_Interpretability/Visualizing_CNN_Filters/Images/block3_conv1_index10.png?raw=true" width="256"  height="256"></code> | <code><img src="https://github.com/MohamedFarag21/ComputerVision-projects/blob/main/CNN_Interpretability/Visualizing_CNN_Filters/Images/block3_conv1_index200.png?raw=true" width="256"  height="256"></code> | <code><img src="https://github.com/MohamedFarag21/ComputerVision-projects/blob/main/CNN_Interpretability/Visualizing_CNN_Filters/Images/block3_conv1_index20.png?raw=true" width="256"  height="256"></code> 

=============

Hope you enjoy it! ♥️
