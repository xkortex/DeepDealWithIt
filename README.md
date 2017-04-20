# Deep DealWithIt

### (•_•)
### ( •_•)>⌐■-■
### (⌐■_■)


This project is based off of the wonderful post by Gabriel Goh, [Decoding the Thought Vector](http://gabgoh.github.io/ThoughtVectors/). My original idea was to train a Variational Autoencoder with [Discriminative Regularization](https://arxiv.org/abs/1602.03220) on the [Celeb A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset along with random images from the internet, then use a classifier to locate "sunglasses thought vector", and apply that to input images. See the sliders under [Facial Hair and Accessories](http://gabgoh.github.io/ThoughtVectors/#morph_angle), third row, to see what I mean.

Unfortunately, I ran into a lot of issues with propagating gradients through very deep VAEs. [This is a known problem with VAEs due to the stochastic layer](https://arxiv.org/abs/1602.02282), and I'm working on getting a Keras implementation of the ladder architecture, but I reckon it won't be ready for the deadline.


## Requirements
- Keras **2.0.2** (there are some [issues](https://github.com/fchollet/keras/issues/5968) with VAE in 2.0.3)
- keras_tqdm - because I'm doing most of my development in Jupyter notebooks, and on my system, the stock Keras progbar likes to freeze my kernel.

## Other Notes
At the moment, this network is a little touchy. It does not always converge on training, so it may need to be re-seeded and trained again.

For reasons beyond me, I find that BatchNorm and Dropout on VAEs tends to *increase* instability at the start of training and may disrupt convergence later in training. Deep learning is still very much a dark art. Reducing the depth of the network and/or reducing the amount of regularization helps convergence (but of course, the trade off here is increased overfitting)

 This model uses [warmup](http://orbit.dtu.dk/files/121765928/1602.02282.pdf), which essentially ignores the divergence of the stochastic layer, allowing the convo layers to "burn in", i.e. converge on feature detection, before trying to force the latent Z into a normal distribution. From what I understand,  putting too much emphasis on the Kullback-Leibler loss term too early can interfere with convergence of the reconstruction term (and you just get ugly blobs for the output).

 I'd really like to use transfer learning to build up something like a deep resnet frontend, layer by layer. Maybe I'll implement that at a later date.