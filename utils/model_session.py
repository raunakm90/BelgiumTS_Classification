# Reference - https://github.com/wpm/tf_model_session/blob/master/model_session.py
import os
import tensorflow as tf


class ModelSession():
    """
    A session of a TensorFlow that may be serialized.
    Overwrite create_graph() to define model graph computation
    """

    def __init__(self, session, saver):
        """
        Create a model session
        :param session: tf.Session() of the model
        :param saver: tf.Saver() that is used to serialize this session
        """
        self.session = session
        self.saver = saver

    @classmethod
    def create(cls, **kwargs):
        """
        Create a new model session
        :param kwargs: optional graph parameters
        :type: dict
        :return: new model session
        :rtype: ModelSession object
        """
        # Start tensorflow session
        session = tf.Session()
        # Create graph in this session
        with session.graph.as_default():
            tf.set_random_seed(147)
            cls.create_graph(**kwargs)
        # Initialize all global variables
        session.run(tf.global_variables_initializer())

        return (cls(session, tf.train.Saver()))

    @staticmethod
    def create_graph(**kwargs):
        """
        Define Tensorflow graph
        :param kwargs: optional graph parameters
        """
        raise NotImplementedError()

    @classmethod
    def restore(cls, checkpoint_dir):
        """
        Restore a serialized model
        :param checkpoint_dir: path to saved checkpoint models
        :return:
        """
        session = tf.Session()
        # Deserialize the graph
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint_file is None:
            raise ValueError("Invalid checkpoint directory %s" % checkpoint_dir)

        # MetaGraph contains both TF GraphDef as well as associated metadata necessary for running computations
        # in a graph when crossing a processs boundary.
        saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
        saver.restore(session, checkpoint_file)

        # Subsequent saves of this model during this session must be done with the same saver object that was used to
        # deserialize it.
        return cls(session, saver)

    def save(self, checkpoint_dir):
        """
        Save the current model session to a checkpoint file.
        If the graph defines an "iteration" variable its value will be used for the global step in the checkpoint name.
        :param checkpoint_dir:  directory containing checkpoint files
        :type checkpoint_dir: str
        :return: path to the new checkpoint file
        :rtype: str
        """

        try:
            iteration = self.session.graph.get_tensor_by_name("iteration:0")
            global_step = self.session.run(iteration)
        except KeyError:
            global_step = None
        path = self.saver.save(self.session, os.path.join(checkpoint_dir, "model.ckpt"), global_step=global_step)
        return path
