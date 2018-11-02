import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from matplotlib import gridspec

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('/tmp/mnist/', one_hot=True)


def weight(shape, x):
    return tf.get_variable(name=x, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias(shape, x):
    return tf.get_variable(name=x, shape=shape, initializer=tf.constant_initializer(0))


# discriminator net

X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = weight([784, 128], 'D_W1')
D_b1 = bias([128], 'D_b1')

D_W2 = weight([128, 1], 'D_W2')
D_b2 = bias([1], 'D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# generator net

Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = weight([100, 128], 'G_W1')
G_b1 = bias([128], 'G_b1')

G_W2 = weight([128, 784], 'G_W2')
G_b2 = bias([784], 'G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


# model

def generator(z):
    x = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    x = tf.matmul(x, G_W2) + G_b2
    x = tf.nn.sigmoid(x)

    return x


def discriminator(x):
    x = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    logit = tf.matmul(x, D_W2) + D_b2
    prob = tf.nn.sigmoid(x)

    return prob, logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels = tf.zeros_like(D_logit_fake)))

D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


def sample_Z(m, n):
    """Uniform prior for G(Z)"""
    return np.random.uniform(-1., 1., size=[m, n])


sess = tf.Session()

sess.run(tf.global_variables_initializer())
for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={
                              X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={
                              Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))


def sample_Z(m, n):
    """Uniform prior for G(Z)"""
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


if not os.path.exists('out/'):
    os.makedirs('out/')

sess.run(tf.global_variables_initializer())

i = 0
for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={
                           Z: sample_Z(16, Z_dim)})  # 16*784
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={
                              X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={
                              Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
