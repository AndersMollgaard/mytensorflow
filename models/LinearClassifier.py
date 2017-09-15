import tensorflow as tf
from ..models.Model import Model

###################################################

class LinearClassifier(Model):
    
    def __init__(self, model_dir='/tmp/LinearClassifier', hparams={'learning_rate': 0.01}):
        Model.__init__(self, model_dir, hparams)
    
    def build_graph(self, data_build_fn=None, data_build_args={}):
        # data ops
        data = data_build_fn(**data_build_args)
        X = data['X']
        Y = data['Y']
        # variables
        input_shape = tf.TensorShape([X.get_shape()[1]])
        weights = tf.Variable(tf.random_normal(input_shape, 0.0, 0.1),
                              name='weights')
        bias = tf.Variable(0., dtype=tf.float32, name='bias')
        # model
        logits = tf.tensordot(X, weights, [[1], [0]]) + bias
        Y_pred = tf.nn.sigmoid(logits, name='probabilities')
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                      labels=Y))
        # summaries
        with tf.name_scope('summaries'):
            tf.summary.scalar("loss", loss)
            tf.summary.histogram("weights", weights)
            summary = tf.summary.merge_all()
        # optimizer
        global_step = tf.Variable(0., trainable=False)
        optimizer = tf.train.AdamOptimizer(self.hparams['learning_rate']).minimize(loss, global_step)
        # evaluation metrics
        classes = tf.greater(logits, tf.zeros_like(logits), name='classes')
        eval_metric_ops = {
            'auc': tf.contrib.metrics.streaming_auc(labels=Y, predictions=Y_pred)[1],
            "accuracy": tf.contrib.metrics.streaming_accuracy(labels=Y, predictions=classes)[1]
            }
        # initialize graph
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init.run()
        # add operations to model
        self.X = X
        self.Y = Y
        self.logits = logits
        self.Y_pred = Y_pred
        self.loss = loss
        self.summary = summary
        self.global_step = global_step
        self.optimizer = optimizer
        self.classes = classes
        self.eval_metric_ops = eval_metric_ops
        