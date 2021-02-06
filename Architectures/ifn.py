import keras
from keras.models import Model
import keras.backend as K
import tensorflow as tf

import numpy as np
import scipy.special
import sklearn.feature_selection

from matplotlib import pyplot as plt
from matplotlib import cm


class IFN(keras.Model):

    # Initialize
    def __init__(self):
        super(IFN, self).__init__()

    def __init__(self, network_x, network_y, network_combined):

        super(IFN, self).__init__()

        # Networks
        self.network_x = network_x
        self.network_y = network_y
        self.network_combined = network_combined

        # # Network Shapes
        # self.x_shape = network_x.layers[0].input_shape
        # self.y_shape = network_y.layers[0].input_shape

        self.means = [0.0, 0.0]
        self.vars = [1.0, 1.0]

    # Neural network function (input given output)
    def feed_forward(self, inputs):
                
        output_x = self.network_x(inputs[0])
        output_y = self.network_y(inputs[1])

        merged = keras.layers.concatenate([output_x, output_y])
        output = self.network_combined(merged)
        return output


    # Run instance of network to get output + derivatives
    def call(self, inputs, training = False):

        if not training:
            # Open 2 gradient tapes to take the second derivative
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(inputs)
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(inputs)
                    output = self.feed_forward(inputs)

                dL_dI = tape1.gradient(output, [inputs[0], inputs[1]])  # First derivatives
            d2L_dI2 = [tape2.gradient(dL_dI[0], [inputs[0], inputs[1]]), tape2.gradient(dL_dI[1], [inputs[0], inputs[1]]),]       # Second derivatives
            return output #, dL_dI, d2L_dI2

        # If training, dont return derivatives
        else:
            return self.feed_forward(inputs)


    # Maximum Likelihood Task
    def maximize(self, input, axis, epochs, samples = None):
        
        # Generate initial random seed
        if samples is None:
            if axis == 0:
                dim = self.x_shape[1]
            elif axis == 1:
                dim = self.y_shape[1]
            N = input.shape(0)
            samples = tf.Variable(tf.random.normal(mean= self.means[axis], stddev = self.vars[axis], shape = (N, dim)))
        else:
            samples = tf.Variable(samples)

        # Train Step
        for epoch in range(epochs):

            with tf.GradientTape() as tape:
                tape.watch(samples)
                out = self([input, samples], training = True) if axis == 0 else self([samples, input], training = True)
                out = out * -1

            gradients = tape.gradient(out, samples)
            self.optimizer.apply_gradients(zip([gradients], [samples]))

        return samples

    def train_step(self, data):
  
        x, y = data[0]
        y_shuffle = tf.random.shuffle(y)

        with tf.GradientTape() as tape:
            out_joint = self([x,y], training=True)  # Forward pass
            out_marginal = self([x,y_shuffle], training=True)  # Forward pass
        
            loss = self.compiled_loss(out_joint, out_marginal, regularization_losses=self.losses)
            # loss += sum(self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(out_joint, out_marginal)
        return {m.name: m.result() for m in self.metrics}

    def train(self, train_dataset, epochs, batch_size = 128, regulate = False):

        x, y = train_dataset[0], train_dataset[1]

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            regulator = 0
            if regulate:
                regulator = max(0, epochs/2 - epoch)

            # Shuffle y  
            y_shuffle = tf.random.shuffle(y)
            for batch in range(int(x.shape[0] / batch_size)):

                current_index = batch_size * batch
                next_index = batch_size * (batch + 1)

                x_batch = x[current_index:next_index]
                y_batch = y[current_index:next_index]
                y_shuffle_batch = y_shuffle[current_index:next_index]

                # Gradients
                with tf.GradientTape() as tape:
                    out_joint = self([x_batch,y_batch], training=True)  # Forward pass
                    out_marginal = self([x_batch,y_shuffle_batch], training=True)  # Forward pass
                
                    loss = self.compiled_loss(out_joint, out_marginal, regularization_losses=self.losses)

                # Compute gradients
                trainable_vars = self.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                # Update weights
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                self.compiled_metrics.update_state(out_joint, out_marginal)
                L = (tf.reduce_mean(out_joint, axis=0) + 1 - tf.reduce_mean(tf.math.exp(out_marginal - regulator), axis = 0))
                L1 = tf.reduce_mean(out_joint, axis=0) + 1
                L2 = tf.reduce_mean(tf.math.exp(out_marginal - regulator), axis = 0)
                print("Epoch %d, Batch %d: Loss = %.3f, Joint = %.3f, Marginal = %0.3f, MI = %.3f, L1 = %.3f, L2 = %.3f, R = %.3f" % (epoch, batch, loss, np.mean(out_joint), np.mean(out_marginal), L, L1, L2, regulator))



    def pre_train(self, train_dataset, epochs, batch_size = 64, verbose = False):
        x, y = train_dataset[0], train_dataset[1]
        sum_x = np.sum(np.square(x), axis= 1)
        sum_y = np.sum(np.square(y), axis= 1)
        norm = np.max(sum_x) + np.max(y)
        s = (sum_x + sum_y) / norm

        # Instantiate an optimizer.
        optimizer = keras.optimizers.Adam()
        # Instantiate a loss function.
        def loss(target_y, predicted_y):
            return tf.reduce_mean(tf.square(target_y - predicted_y))

        L = 1
        prev_MI_1 = 1
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            for batch in range(int(x.shape[0] / batch_size)):

                current_index = batch_size * batch
                next_index = batch_size * (batch + 1)

                x_batch = x[current_index:next_index]
                y_batch = y[current_index:next_index]
                y_shuffle = tf.random.shuffle(y_batch)

                prev_MI_2 = prev_MI_1
                prev_MI_1 = L

                with tf.GradientTape() as tape:   

                    output_joint = self([x_batch, y_batch], training = True)
                    output_marginal = self([x_batch, y_shuffle], training = True)
                    loss_value =  tf.reduce_mean(tf.square(output_joint))+ tf.reduce_mean(tf.square(output_marginal)) 
                    # loss_value = tf.reduce_mean(tf.math.exp(output_marginal), axis = 0)

                grads = tape.gradient(loss_value, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
                L = (tf.reduce_mean(output_joint, axis=0) + 1 - tf.reduce_mean(tf.math.exp(output_marginal), axis = 0))
                L1 = tf.reduce_mean(output_joint, axis=0) + 1
                L2 = tf.reduce_mean(tf.math.exp(output_marginal), axis = 0)
                if verbose: 
                    print("Epoch %d, Batch %d: Loss = %.3f, Joint = %.3f, Marginal = %0.3f, MI = %.3f" % (epoch, batch, loss_value, np.mean(output_joint), np.mean(output_marginal), L))

                # if np.abs(L) < 0.01 and np.abs(prev_MI_1) < 0.01 and np.abs(prev_MI_2) < 0.01:
                #     return




class gIFN(IFN):

    # Initialize
    def __init__(self, network_A, network_B, network_C, network_D):

        super(gIFN, self).__init__(network_A, network_B, network_C)

        # Networks
        self.network_A = network_A
        self.network_B = network_B
        self.network_C = network_C
        self.network_D = network_D


        self.means = [0.0, 0.0]
        self.vars = [1.0, 1.0]

    # Neural network function (input given output)
    def feed_forward(self, inputs):
                
        output_A = self.network_A(inputs[1])
        output_B = self.network_B(inputs[1])
        output_C = self.network_C([inputs[0], inputs[1]])
        output_C_symmetrized = output_C
        output_D = self.network_D(inputs[1])


        difference = inputs[0] - output_B
        matmul = (tf.matmul(output_C_symmetrized , difference[...,None]))[...,0]

        output = output_A + 1.0*tf.keras.layers.Dot(axes = 1)([difference, output_D]) + 0.5 * tf.keras.layers.Dot(axes = 1)([difference, matmul ])
        # out = tf.math.log(tf.math.softplus(output))
        return output


    def ensemble_maximum(self, y_ensemble):

#         c = self.network_C.predict(np.ravel(y_ensemble)).reshape(y_ensemble.s)
#         mesh_z = np.array(g(np.ravel(mesh_x), np.ravel(mesh_y)))
# mesh_z = mesh_z.reshape(mesh_x.shape)

        b = self.network_B.predict(y_ensemble)
        c = self.network_C.predict([b, y_ensemble])
        cov = tf.linalg.inv(tf.reduce_sum(c, 0))

        weighted_b = tf.reduce_sum(tf.matmul(c, b[...,None])[...,0], 0)
        x = tf.matmul(cov, weighted_b[...,None])[...,0]
        
        return x.numpy(), -1*cov.numpy()


    def get_network_A(self):
        return self.network_A

    def get_network_B(self):
        return self.network_B

    def get_network_C(self):
        return self.network_C

    def maximum_likelihood(self, x):
        return self.network_B.predict(x)

    def covariance(self, x):
        b = self.maximum_likelihood(x)
        b = np.squeeze(b)
        print(b.shape)
        return -1 * tf.linalg.inv(self.network_C.predict([x, b])).numpy()

def f_loss(out_joint, out_marginal):
    return -(tf.reduce_mean(out_joint, axis=0) - tf.reduce_mean(tf.math.exp(out_marginal - 1), axis = 0))

def mine_loss(out_joint, out_marginal):
    return -(tf.reduce_mean(out_joint, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(out_marginal), axis = 0)))

def reset_weights(model):
    print("Reset")
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))
            print('reinitializing layer {}'.format(layer.name))

