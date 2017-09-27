from distutils.version import LooseVersion
import warnings
import problem_unittests as tests
import tensorflow as tf
import numpy as np

class dcgan():
    
    def __init__(self, bsize, lrate, zdim, beta1, epochs):
        """
        Initialization
        :param epochs: Number of epochs
        :param bsize: Batch Size
        :param zdim: Z dimension
        :param lrate: Learning Rate
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        """
        self.batch_size = bsize
        self.z_dim = zdim
        self.learning_rate = lrate
        self.beta1 = beta1
        self.epochs = epochs
        
    def check_gpu(self):
        print('This will check to make sure you have the correct version of TensorFlow and access to a GPU')
        
        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
        print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        
        
    def unit_test(self):
        tests.test_model_inputs(self.model_inputs)
        tests.test_discriminator(self.discriminator, tf)
        tests.test_generator(self.generator, tf)
        tests.test_model_loss(self.model_loss)
        tests.test_model_opt(self.model_opt, tf)
        
    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        """
        Create the model inputs
        :param image_width: The input image width
        :param image_height: The input image height
        :param image_channels: The number of image channels
        :param z_dim: The dimension of Z
        :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
        """
        input_real = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name='input_real')
        input_z = tf.placeholder(tf.float32, [None, z_dim], name='input_z')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        return input_real, input_z, learning_rate

    def discriminator(self, images, reuse=False):
        """
        Create the discriminator network
        :param image: Tensor of input image(s)
        :param reuse: Boolean if the weights should be reused
        :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
        """
        alpha = 0.1
    
        with tf.variable_scope('discriminator', reuse=reuse):
            conv1 = tf.layers.conv2d(images, 64, 3, strides=2, padding='same')
            conv1 = tf.maximum(alpha*conv1, conv1)
        
            conv2 = tf.layers.conv2d(conv1, 128, 3, strides=2, padding='same')
            conv2 = tf.layers.batch_normalization(conv2, training=True)
            #conv2 = tf.nn.relu(conv2)
            conv2 = tf.maximum(alpha*conv2, conv2)
        
            conv3 = tf.layers.conv2d(conv2, 256, 3, strides=2, padding='same')
            conv3 = tf.layers.batch_normalization(conv3, training=True)
            #conv3 = tf.nn.relu(conv3)
            conv3 = tf.maximum(alpha*conv3, conv3)
        
            size = conv3.get_shape().as_list()
            n = size[1] * size[2] * size[3]
            flat = tf.reshape(conv3, [-1, n])
            logits = tf.layers.dense(flat, 1, activation=None)
            output = tf.sigmoid(logits)
        return output, logits

    def generator(self, z, out_channel_dim, is_train=True):
        """
        Create the generator network
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """
        alpha = 0.1
        with tf.variable_scope('generator', reuse=not is_train):
            x1 = tf.layers.dense(z, 3 * 3 * 256)
            x1 = tf.reshape(x1, [-1, 3, 3, 256])    
            #x1 = tf.layers.batch_normalization(x1, training=is_train)
            #x1 = tf.nn.relu(x1)
            x1 = tf.maximum(alpha*x1, x1)
        
            #x2 = tf.image.resize_nearest_neighbor(x1, (7,7))
            #x2 = tf.layers.conv2d(x2, 256, 5, padding='same')
            x2 = tf.layers.conv2d_transpose(x1, 128, 3, strides=2, padding='valid')
            x2 = tf.layers.batch_normalization(x2, training=is_train)
            #x2 = tf.nn.relu(x2)
            x2 = tf.maximum(alpha*x2, x2)
        
            x3 = tf.layers.conv2d_transpose(x2, 64, 3, strides=2, padding='same')
            x3 = tf.layers.batch_normalization(x3, training=is_train)
            #x3 = tf.nn.relu(x3)
            x3 = tf.maximum(alpha*x3, x3)
        
            logits = tf.layers.conv2d_transpose(x3, out_channel_dim, 3, strides=2, padding='same', activation=None)
            logits = tf.tanh(logits)

        return logits
    
    
    def model_loss(self, input_real, input_z, out_channel_dim):
        """
        Get the loss for the discriminator and generator
        :param input_real: Images from the real dataset
        :param input_z: Z input
        :param out_channel_dim: The number of channels in the output image
        :return: A tuple of (discriminator loss, generator loss)
        """
        g_model = self.generator(input_z, out_channel_dim, is_train=True)
 
        d_model_real, d_logits_real = self.discriminator(input_real, reuse=False)
        d_model_fake, d_logits_fake = self.discriminator(   g_model, reuse=True)

        d_label_real = tf.ones_like(d_model_real) * 0.9
        d_label_fake = tf.zeros_like(d_model_fake)
        g_label_fake = tf.ones_like(d_model_fake)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=d_label_real))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=d_label_fake))
        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=g_label_fake))
        return d_loss, g_loss
    
    
    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (discriminator training operation, generator training operation)
        """
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt


    def show_generator_output(self, sess, n_images, input_z, out_channel_dim, image_mode):
        """
        Show example output for the generator
        :param sess: TensorFlow session
        :param n_images: Number of Images to display
        :param input_z: Input Z Tensor
        :param out_channel_dim: The number of channels in the output image
        :param image_mode: The mode to use for images ("RGB" or "L")
        """
        cmap = None if image_mode == 'RGB' else 'gray'
        z_dim = input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

        samples = sess.run(self.generator(input_z, out_channel_dim, False), feed_dict={input_z: example_z})

        images_grid = helper.images_square_grid(samples, image_mode)
        pyplot.imshow(images_grid, cmap=cmap)
        pyplot.show()
        
        
    def train(self, get_batches, data_shape, data_image_mode, save_path='./checkpoints/generator.ckpt'):
        """
        Train the GAN
        :param get_batches: Function to get batches
        :param data_shape: Shape of the data
        :param data_image_mode: The image mode to use for images ("RGB" or "L")
        """

        image_channels = data_shape[3]
    
        input_real, input_z, _ = self.model_inputs(data_shape[1], data_shape[2], image_channels, self.z_dim)
    
        d_loss, g_loss = self.model_loss(input_real, input_z, image_channels)
        d_opt, g_opt = self.model_opt(d_loss, g_loss, self.learning_rate, self.beta1)
    
        step = 0
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(self.epochs):
                for batch_images in get_batches(self.batch_size):
                    step += 1
                    batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                    _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z}) 
                    _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                    _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                    _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                    
                    if step % 50 == 0:
                        d_loss_out = sess.run(d_loss, feed_dict={input_real: batch_images, input_z: batch_z})
                        g_loss_out = sess.run(g_loss, feed_dict={input_z: batch_z})
                        print("Epoch {}/{}...".format(epoch_i+1, self.epochs),
                              "Discriminator Loss: {:.4f}...".format(d_loss_out),
                                     "Generator Loss: {:.4f}".format(g_loss_out))
                    if step % 500 == 0:
                        self.show_generator_output(sess, 10, input_z, image_channels, data_image_mode)
                        
        print("Finished training ...")
        saver = tf.train.Saver()
        saver.save(sess, save_path)