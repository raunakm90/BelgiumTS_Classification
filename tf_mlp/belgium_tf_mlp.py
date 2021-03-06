import sys

sys.path.append("D:/ML_Projects/BelgiumTS")
import tensorflow as tf
from data.belgiumTS import load_BelgiumTS
from utils.model_session import ModelSession
import numpy as np
from utils.generate_report import plot_confusion_matrix

# tf.flags is a thin wrapper around argparse.
# Define all hyper-parameters as FLAGS.

# Data Loading parameters
tf.flags.DEFINE_float(flag_name="val_percentage", default_value=.1,
                      docstring="Percentage of the training data to use for validation")
tf.flags.DEFINE_integer(flag_name="image_size", default_value=28, docstring="Resize image shape")
tf.flags.DEFINE_integer(flag_name="num_classes", default_value=62, docstring="Number of classes defined in the dataset")

# Model Hyperparameters
tf.flags.DEFINE_float(flag_name="learning_rate", default_value=0.001, docstring="Learning Rate for optimization")
tf.flags.DEFINE_integer(flag_name="layer1_units", default_value=784, docstring="Number of units layer 1")
tf.flags.DEFINE_integer(flag_name="layer2_units", default_value=196, docstring="Number of units layer 2")
tf.flags.DEFINE_integer(flag_name="layer3_units", default_value=49, docstring="Number of units layer 3")

# Training Parameters
tf.flags.DEFINE_integer(flag_name="batch_size", default_value=60, docstring="Batch Size (default: 60)")
tf.flags.DEFINE_integer(flag_name="nb_epochs", default_value=100,
                        docstring="Number of training epochs/iterations (default: 100")

# Tensorboard Parameters
tf.flags.DEFINE_string(flag_name="log_dir", default_value="./tf_mlp/log_dir/",
                       docstring="Log directory for storing model performance")
tf.flags.DEFINE_bool(flag_name="clean_log", default_value=True, docstring="Clean log files")
tf.flags.DEFINE_string(flag_name="model_dir", default_value="./tf_mlp/model_dir/",
                       docstring="Path to save model as well as checkpoints")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


class BelgiumTS_NN(ModelSession):
    """
    Model Session that implements a simple neural network from scratch
    on the Belgium Traffic Signals classification dataset.
    """

    @staticmethod
    def create_graph(layer_1, layer_2, layer_3):
        def weight_variable(shape):
            """weight_variable generates a weight variable of a given shape."""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape, name=None):
            """bias_variable generates a bias variable of a given shape."""
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        # Keeps track of iterations in the graph across calls to run()
        iteration = tf.Variable(initial_value=0, trainable=False, name="iteration")

        # Difference between tf.variable_scope() and tf.name_scope()
        # tf.variable_scope() adds a prefix to the names of all variables
        # tf.name_scope() ignores variables created with tf.get_variable()

        # Variable scope that reads data and model parameters
        with tf.variable_scope("parameters"):
            x = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size * FLAGS.image_size], name="x")
            y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name="y")
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # Variable scope to Layer 1
        with tf.variable_scope("layer_1"):
            w_layer1 = weight_variable(shape=[FLAGS.image_size * FLAGS.image_size, layer_1])
            b_layer1 = bias_variable(shape=[layer_1], name="bias")
            h_layer1 = tf.nn.relu(tf.matmul(x, w_layer1) + b_layer1)
            # variable_summaries(w_layer1)

        # Variable scope to Layer 2
        with tf.variable_scope("layer_2"):
            w_layer2 = weight_variable(shape=[layer_1, layer_2])
            b_layer2 = bias_variable(shape=[layer_2], name="bias")
            h_layer2 = tf.nn.relu(tf.matmul(h_layer1, w_layer2) + b_layer2)
            # variable_summaries(w_layer2)

        # Variable scope to Layer 3
        with tf.variable_scope("layer_3"):
            w_layer3 = weight_variable(shape=[layer_2, layer_3])
            b_layer3 = bias_variable(shape=[layer_3], name="bias")
            h_layer3 = tf.nn.relu(tf.matmul(h_layer2, w_layer3) + b_layer3)
            # variable_summaries(w_layer1)

        # Variable scope to Output Layer
        with tf.variable_scope("output_layer"):
            w_output_layer = weight_variable(shape=[layer_3, FLAGS.num_classes])
            b_output_layer = bias_variable(shape=[FLAGS.num_classes], name="bias")
            y_logits = tf.matmul(h_layer3, w_output_layer) + b_output_layer
            # variable_summaries(w_output_layer)
            tf.summary.histogram('logits', y_logits)

        # Variable scope to train neural network
        with tf.variable_scope("train"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_logits))
            tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=iteration,
                                                                         name="train_step")
        tf.summary.scalar('Loss', cross_entropy)

        # Evaluation
        with tf.variable_scope("evaluation"):
            correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
            y_pred = tf.argmax(y_logits, 1, name='predictions')
            tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    def __str__(self):
        return "BelgiumTS Simple NN (Layer 1 Units: %d, Layer 2 Units: %d, Layer 3 Unites:%d, Iteration %d)" % (
            self.hidden_layer_1.get_shape()[0], self.hidden_layer_2.get_shape()[0], self.hidden_layer_3.get_shape()[0],
            self.session.run(self.iteration)
        )

    def train(self, x, y, learning_rate, merged):
        """
        Train model using train_step defined in our graph definition
        :param merged: tf.summary.merge_all()
        :param x: Input images
        :param y: Labels
        :param learning_rate: Learning rate for optimization
        :return: Iteration number
        """
        summary, _, step = self.session.run([merged, self.train_step, self.iteration],
                                            feed_dict={self.x: x,
                                                       self.y: y,
                                                       self.learning_rate: learning_rate})

        return summary, step

    def eval(self, x, y, merged):
        """
        Evaluate model performance based on the trained object
        :param x: Images
        :param y: Labels
        :param merged: tf.summary.merge.all()
        :return: accuracy
        """
        summary, accuracy = self.session.run([merged, self.accuracy],
                                             feed_dict={self.x: x,
                                                        self.y: y})
        return summary, accuracy

    def test(self, x, y):
        """
        Evaluate model performance based on the trained object
        :param x: Images
        :param y: Labels
        :return: accuracy
        """
        accuracy, preds = self.session.run([self.accuracy, self.predictions],
                                           feed_dict={self.x: x,
                                                      self.y: y})
        return accuracy, preds

    def _tensor(self, name):
        return self.session.graph.get_tensor_by_name(name)

    @property
    def hidden_layer_1(self):
        return self._tensor("layer_1/bias:0")

    @property
    def predictions(self):
        return self._tensor("evaluation/predictions:0")

    @property
    def hidden_layer_2(self):
        return self._tensor("layer_2/bias:0")

    @property
    def hidden_layer_3(self):
        return self._tensor("layer_3/bias:0")

    @property
    def train_step(self):
        return self._tensor("train/train_step:0")

    @property
    def x(self):
        return self._tensor("parameters/x:0")

    @property
    def y(self):
        return self._tensor("parameters/y:0")

    @property
    def learning_rate(self):
        return self._tensor("parameters/learning_rate:0")

    @property
    def iteration(self):
        return self._tensor("iteration:0")

    @property
    def accuracy(self):
        return self._tensor("evaluation/accuracy:0")


def clean_log_files(log_dir_path):
    """
    Clean log files for the log directory
    :param log_dir_path: path to log directory
    :return: None
    """
    if tf.gfile.Exists(log_dir_path):
        tf.gfile.DeleteRecursively(log_dir_path)
    tf.gfile.MakeDirs(log_dir_path)


def train(training_data, validation_data):
    model = BelgiumTS_NN.create(layer_1=FLAGS.layer1_units, layer_2=FLAGS.layer2_units, layer_3=FLAGS.layer3_units)
    # print(model)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + "train", model.session.graph)
    val_writer = tf.summary.FileWriter(FLAGS.log_dir + "val", model.session.graph)

    for i in range(FLAGS.nb_epochs):
        summary, valid_accuracy = model.eval(validation_data.images, validation_data.labels, merged)
        val_writer.add_summary(summary, i)
        print("%s: Validation Accuracy %0.4f" % (model, valid_accuracy))
        for _ in range(training_data.num_examples // FLAGS.batch_size):
            x, y = training_data.next_batch(FLAGS.batch_size)
            summary, iteration = model.train(x, y, FLAGS.learning_rate, merged)
            # train_writer.add_summary(summary, iteration)
        train_writer.add_summary(summary, iteration)
    val_writer.close()
    train_writer.close()
    # Save Final Model
    model.save(FLAGS.model_dir)
    print("Final Model: ", model)


def test(testing_data, merged):
    model = BelgiumTS_NN.restore(FLAGS.model_dir)
    # Evaluate on test data
    test_accuracy, test_preds = model.test(testing_data.images, testing_data.labels)
    np.savez("./tf_mlp/Test_Preds.npz", test_preds)
    print("%s: Testing Accuracy %0.4f" % (model, test_accuracy))
    return test_preds


def main(argv=None):
    belgiumTS_data = load_BelgiumTS(validation_size=FLAGS.val_percentage, resize_pix=FLAGS.image_size)
    training_data = belgiumTS_data.train
    validation_data = belgiumTS_data.validation
    testing_data = belgiumTS_data.test
    if FLAGS.clean_log:
        clean_log_files(FLAGS.log_dir)
        clean_log_files(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir + "train")
    tf.gfile.MakeDirs(FLAGS.log_dir + "val")

    print("Training model parameters")
    train(training_data, validation_data)
    print("Evaluating on Testing data")
    test_pred = test(testing_data, None)
    plot_confusion_matrix(y_true=np.argmax(testing_data.labels, axis=1), y_pred=test_pred,
                          file_name="./tf_mlp/figures/Confusion_Matrix.png")


if __name__ == '__main__':
    # The tf.app.run() invocation that ensures that any flags are parsed, and then invokes the main() function in the same module.
    tf.app.run()
