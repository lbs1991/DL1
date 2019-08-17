import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#loading dataset
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#number per bunch
batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#creat a neureal network
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros[10])
P = tf.nn.softmax(tf.matnul(x,W)+b)

loss = tf.reduce_mean(tf.square(y-P))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#help(tf.argmax)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(P,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sees.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy "+ str(acc))