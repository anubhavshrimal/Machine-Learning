import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
# one_hot gives labels as:
# 0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] and so on

# 28x28 = 784 pixel images of numbers
mnist = input_data.read_data_sets('../../datasets/mnist/', one_hot=True)

# epochs = number of cycles = forward propagation + back propagation
epochs = 3
# Number of output nodes
n_classes = 10
# Number of images fed into the Neural Network at a time
batch_size = 128

# Set size of chunks and number of chunks to pass in the RNN
chunk_size = 28
n_chunks = 28

rnn_size = 128

# shape of the input data and output labels
# shape parameter is optional
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def rnn_model(x):
    layer = {
        'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output


def train_nn(x, y):
    prediction = rnn_model(x)

    # Find the cost or error in prediction made and labels
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Reduce the cost using an optimizer
    # default learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train the model
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # Get data and labels in batch_size
                x_, y_ = mnist.train.next_batch(batch_size)

                x_ = x_.reshape((batch_size, n_chunks, chunk_size))

                # Run the optimizer to minimize the cost = c
                _, c = sess.run([optimizer, cost], feed_dict={x: x_, y: y_})
                epoch_loss += c
            print('Epoch', (epoch + 1), 'completed out of:', epochs, 'loss:', epoch_loss)

        # Test the model accuracy
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)),
                                          y: mnist.test.labels}))

train_nn(x, y)
