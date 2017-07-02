import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# one_hot gives labels as:
# 0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] and so on

# 28x28 = 784 pixel images of numbers
mnist = input_data.read_data_sets('../../datasets/mnist/', one_hot=True)

# Number of nodes in each hidden layer
nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500
# Number of output nodes
n_classes = 10

# Number of images fed into the Neural Network at a time
batch_size = 100

# shape of the input data and output labels
# shape parameter is optional
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def nn_model(data):
    hiddent_layer_1 = {
        'weights': tf.Variable(tf.truncated_normal([784, nodes_hl1], stddev=0.1)),
        'biases': tf.Variable(tf.constant(0.1, shape=[nodes_hl1]))
    }

    hiddent_layer_2 = {
        'weights': tf.Variable(tf.truncated_normal([nodes_hl1, nodes_hl2], stddev=0.1)),
        'biases': tf.Variable(tf.constant(0.1, shape=[nodes_hl2]))
    }

    hiddent_layer_3 = {
        'weights': tf.Variable(tf.truncated_normal([nodes_hl2, nodes_hl3], stddev=0.1)),
        'biases': tf.Variable(tf.constant(0.1, shape=[nodes_hl3]))
    }

    output_layer = {
        'weights': tf.Variable(tf.truncated_normal([nodes_hl3, n_classes], stddev=0.1)),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_classes]))
    }

    # output_for_layer = (input * weights) + biases
    # pass the layer_outputs to the activation function; we used relu in this case
    l1 = tf.add(tf.matmul(data, hiddent_layer_1['weights']), hiddent_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hiddent_layer_2['weights']), hiddent_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hiddent_layer_3['weights']), hiddent_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


def train_nn(x, y):
    prediction = nn_model(x)

    # Find the cost or error in prediction made and labels
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Reduce the cost using an optimizer
    # default learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # epochs = number of cycles = forward propagation + back propagation
    epochs = 10
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.isfile('./checkpoint'):
            print('checkpoint restored')
            saver.restore(sess, "mnist_nn.ckpt")

        # Train the model
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # Get data and labels in batch_size
                x_, y_ = mnist.train.next_batch(batch_size)
                # Run the optimizer to minimize the cost = c
                _, c = sess.run([optimizer, cost], feed_dict={x: x_, y: y_})
                epoch_loss += c
            print('Epoch', (epoch + 1), 'completed out of:', epochs, 'loss:', epoch_loss)

            # Save the model in checkpoint after each epoch
            save_path = saver.save(sess, "mnist_nn.ckpt")
            print("Model saved in file: %s" % save_path)

        # Test the model accuracy
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_nn(x, y)
