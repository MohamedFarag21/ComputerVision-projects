# %% Import required modules
import tensorflow as tf
import keras

# %% Choose your model
model = tf.keras.applications.VGG16(
weights="imagenet",
include_top=False)

# %% Slice your model based on the layer you want to visualize its filters' pattern

layer_name = "block3_conv1"
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)
feature_extractor.summary()

# %% Create a loss function to detect the measure the response of the filter
"""From here, the code is provided by Francois Chollet"""

def compute_loss(image, filter_index):
  activation = feature_extractor(image)
  filter_activation = activation[:, 2:-2, 2:-2, filter_index]
  return tf.reduce_mean(filter_activation)

# %% Use Gradient-ascent in order to modify pixel values of the input image

@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
  with tf.GradientTape() as tape:
    tape.watch(image)
    loss = compute_loss(image, filter_index)
  grads = tape.gradient(loss, image)
  grads = tf.math.l2_normalize(grads)
  image += learning_rate * grads
  return image

# %%
img_width = 200
img_height = 200
def generate_filter_pattern(filter_index):
  iterations = 30
  learning_rate = 10.
  image = tf.random.uniform(
  minval=0.4,
  maxval=0.6,
  shape=(1, img_width, img_height, 3))
  for i in range(iterations):
    image = gradient_ascent_step(image, filter_index, learning_rate)
  return image[0].numpy()

# %% deprocess the image
def deprocess_image(image):
  image -= image.mean()
  image /= image.std()
  image *= 64
  image += 128
  image = np.clip(image, 0, 255).astype("uint8")
  image = image[25:-25, 25:-25, :]
  return image

# %% Plot the image
plt.axis("off")
plt.imshow(deprocess_image(generate_filter_pattern(filter_index=10)))

