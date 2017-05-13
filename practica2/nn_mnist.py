import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


# ---------------- Visualizing some element of the MNIST dataset --------------
""""
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[57]
"""


# D I V I D I E N D O  E L  C O N J U N T O

# Conjunto de entrenamiento
y_entrenamiento_data = one_hot(train_y.astype(int), 10)  # the labels are in the last row. Then we encode them in one hot code

# Conjunto de validacion
y_conjunto_validacion = one_hot(valid_y.astype(int), 10)

# Conjunto test
y_conjunto_test = one_hot(test_y.astype(int), 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

# miEntrenamiento
error = 3.45e9
error_anterior = 50
epoca = 0
print "Error", error, "Error anterior", error_anterior

while error > error_anterior:
    if error < 0.3: break

    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_entrenamiento_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    if epoca > 25:
        error = error_anterior

    error_anterior = sess.run(loss, feed_dict={x: valid_x, y_: y_conjunto_validacion})

    # print "Error", error, "Error anterior", error_anterior
    # print "Epoch #:", epoca, "Error: ",sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    print error_anterior
    epoca+=1
    result = sess.run(y, feed_dict={x: valid_x})
    '''
    for b, r in zip(batch_ys, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"
    '''
print "Start testing...  "
fallo = 0
l = []
count = 0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(y_conjunto_test, result):
    if np.argmax(b) != np.argmax(r):
        fallo +=1
        l.append(count)
    #print "Iteracion: ",count, b, "-->", r
    count+=1
print "Numero de fallos totales: ", fallo, ", Posicion de fallo/s: ", l
print "----------------------------------------------------------------------------------"
