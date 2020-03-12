import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib

path_to_image = 'images/dog.jpg'
path_to_hacked = 'hacked/dog.jpg'
#ACTUAL_CLASS =207
WANTED_CLASS = 852
EPOCHS = 2
LEARNING_RATE = .01
LOSS = []

def load_image(path_to_image):
    max_dim = 512
    img = tf.io.read_file(path_to_image)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    #shape = tf.cast(tf.shape(img)[:-1],tf.float32)
    #long_dim = max(shape)
    #scale = max_dim /long_dim
    #new_shape = tf.cast(shape*scale,tf.int32)
    #img = tf.image.resize(img,new_shape)
    return img[tf.newaxis,:]

def imshow(image,title=None):
    if len(image.shape) > 3:
      image=tf.squeeze(image)
    plt.imshow(image)

    if title:
        plt.title(title)


def loss(model,processedImage):
    y = model(processedImage)[0][WANTED_CLASS]
    return -y

def gradient(model,image):
    with tf.GradientTape() as t :
        processedImage = tf.keras.applications.vgg19.preprocess_input(image*255)
        l = loss(model,processedImage)
        LOSS.append(l.numpy())
        return t.gradient(l,image)

def train(model,image):
   opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
   for epoch in range(EPOCHS):
       print(epoch)
       gr = gradient(model,image)
       opt.apply_gradients([(gr,image)])
       image.assign(clip_0_1(image))

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def plot_images(image,image0):
    plt.subplots(1,2)
    plt.subplot(1,2,1)
    imshow(image0)
    plt.subplot(1,2,2)
    imshow(image)
    plt.show()

image = load_image(path_to_image)
image = tf.image.resize(image,(244,244))
image = tf.Variable(image)
image0 = tf.Variable(image.initialized_value())
#image0 = load_image('hacked/cat.jpg')
#image0 = tf.image.resize(image0,(244,244))

vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
train(vgg,image)
#print([-l for l in LOSS])
prediction = tf.keras.applications.vgg19.decode_predictions(vgg(tf.keras.applications.vgg19.preprocess_input(image*255)).numpy())
print(prediction)
matplotlib.image.imsave(path_to_hacked, image.numpy()[0])
plot_images(image,image0)


#print(tf.math.argmax(vgg(image),axis=1))"""
