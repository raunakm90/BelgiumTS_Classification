import warnings
warnings.filterwarnings("ignore")
from data import load
from skimage.color import rgb2gray
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

RESIZE_PIX = 28


def read_data():
    data_obj = load.data_processing()
    train_imgs, train_labels = data_obj.training_data()
    test_imgs, test_labels = data_obj.testing_data()

    print("Resize images")
    train_imgs = data_obj.resize_imgs(train_imgs, (RESIZE_PIX, RESIZE_PIX))
    test_imgs = data_obj.resize_imgs(test_imgs, (RESIZE_PIX, RESIZE_PIX))

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.transform(test_labels)

    return train_imgs, train_labels, test_imgs, test_labels


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def nn_model(x):
    # Layer 1 weights
    w_layer1 = weight_variable(shape=[784, 10])
    b_layer1 = bias_variable(shape=[10])

    # Layer 2 weights
    w_layer2 = weight_variable(shape=[10, 62])
    b_layer2 = bias_variable(shape=[NUM_CLASSES])

    input_layer = tf.matmul(x, w_layer1) + b_layer1
    hidden_layer = tf.nn.relu(input_layer)
    output_layer = tf.matmul(hidden_layer, w_layer2) + b_layer2

    return output_layer


def run_nn_model():
    global NUM_CLASSES
    x_train, y_train, x_test, y_test = read_data()
    NUM_CLASSES = y_train.shape[1]

    rgb_gray = True

    if rgb_gray:
        x_train = rgb2gray(x_train)
        x_test = rgb2gray(x_test)

    print("Training images: ", x_train.shape)
    print("Training labels: ", y_train.shape)

    batch_size = 512

    # Flatten out images
    x_train = x_train.reshape(-1, RESIZE_PIX*RESIZE_PIX)
    x_test = x_test.reshape(-1, RESIZE_PIX*RESIZE_PIX)


    # Define placeholders for x and y
    x = tf.placeholder(tf.float32, [None, RESIZE_PIX*RESIZE_PIX])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    y_nn = nn_model(x)

    # Loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
    correct_predictions = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10):
            offset = (step*batch_size) % (y_train.shape[0] - batch_size)
            minibatch_data = x_train[offset:(offset+batch_size),:]
            minibatch_labels = y_train[offset:(offset+batch_size)]
            train_step.run(feed_dict={x:minibatch_data, y_:minibatch_labels})

            train_accuracy = accuracy.eval(feed_dict={x: minibatch_data, y_: minibatch_labels})
            print("Step %d, training accuracy %g" % (step, train_accuracy))

        print("Test accuracy %g" % accuracy.eval(feed_dict={x:x_test, y_:y_test}))


if __name__ == '__main__':
    run_nn_model()