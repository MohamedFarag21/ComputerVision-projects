# Visualizing Convnet filters

*The aim is to understand what is the pattern at the image that a CNN filter responds to ?*

# How can we do it ?

1. Generate a random image.
2. Pass the image to the model.
3. Get the filter you want to understand the pattern it encodes.
4. Get the output.
5. Measure if the filter response is totaly maximized.
6. Utilize _Gradient_ascent_ to maximize the response of the filter.
7. Back propagtion is used to modify the pixel values to find "**The image that will maximize the filter's response**"
