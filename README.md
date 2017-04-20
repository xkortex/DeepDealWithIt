# Deep DealWithIt

### (•_•)
### ( •_•)>⌐■-■
### (⌐■_■)


This project is based off of the wonderful post by Gabriel Goh, [Decoding the Thought Vector](http://gabgoh.github.io/ThoughtVectors/). My original idea was to train a Variational Autoencoder with [Discriminative Regularization](https://arxiv.org/abs/1602.03220) on the [Celeb A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset along with random images from the internet, then use a classifier to locate "sunglasses thought vector", and apply that to input images. See the sliders under [Facial Hair and Accessories](http://gabgoh.github.io/ThoughtVectors/#morph_angle), third row, to see what I mean.

Unfortunately, I ran into a lot of issues with propagating gradients through very deep VAEs. [This is a known problem with VAEs due to the stochastic layer](https://arxiv.org/abs/1602.02282), and I'm working on getting a Keras implementation of the ladder architecture, but I reckon it won't be ready for the deadline.


## Requirements
- Keras **2.0.2** (there are some [issues](https://github.com/fchollet/keras/issues/5968) with VAE in 2.0.3)