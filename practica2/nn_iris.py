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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

# O R I G I N A L
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

# D I V I D I E N D O  E L  C O N J U N T O

# Conjunto de entrenamiento
x_entrenamiento_data = data[:105, 0:4].astype('f4')  # the samples are the four first rows of data
y_entrenamiento_data = one_hot(data[:105, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

# Conjunto de validacion
x_conjunto_validacion = data[105:129, 0:4].astype('f4')
y_conjunto_validacion = one_hot(data[105:129, 4].astype(int), 3)

# Conjunto test
x_conjunto_test = data[129:150, 0:4].astype('f4')
y_conjunto_test = one_hot(data[129:151, 4].astype(int), 3)

# Imprime ejemplos de la operacion realizada
x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

# miEntrenamiento
error = 3.45e9
error_anterior = 50;
epoca = 0
print "Error", error, "Error anterior", error_anterior

while error > error_anterior:
    if error < 0.5: break

    for jj in xrange(len(x_entrenamiento_data) / batch_size):
        batch_xs = x_entrenamiento_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_entrenamiento_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    if epoca > 25:
        error = error_anterior

    #error_anterior = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
	error_anterior = sess.run(loss, feed_dict={x: x_conjunto_validacion, y_: y_conjunto_validacion})
    #print sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
	print sess.run(loss, feed_dict={x: x_conjunto_validacion, y_: y_conjunto_validacion})
    #print "Epoch #:", epoca, "Error: ",sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    epoca+=1
    result = sess.run(y, feed_dict={x: x_conjunto_validacion})

print "Start testing...  "
fallo = 0
l = []
count = 0
result = sess.run(y, feed_dict={x: x_conjunto_test})
for b, r in zip(y_conjunto_test, result):
    if np.argmax(b) != np.argmax(r):
        fallo +=1
        l.append(count)
    #print "Iteracion: ",count, b, "-->", r
    count+=1
print "Numero de fallos totales: ", fallo, ", Posicion de fallo/s: ", l
print "----------------------------------------------------------------------------------"
