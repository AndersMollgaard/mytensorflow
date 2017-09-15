import tensorflow as tf
from ..models.Model import Model

###################################################

class LSTMClassifier(Model):
    
    def __init__(self, model_dir='/tmp/LSTMClassifier', 
                 hparams={'learning_rate': 0.01, 'n_hiddens': [30]}):
        Model.__init__(self, model_dir, hparams)
    
    def build_graph(self, data_build_fn=None, data_build_args={}):
        # data ops
        data = data_build_fn(**data_build_args)
        X = data['X']
        Y = data['Y']
        # model
        n_hiddens = self.hparams['n_hiddens']
        cells = [tf.contrib.rnn.BasicLSTMCell(n_hidden) for n_hidden in n_hiddens]
        rnn = tf.contrib.rnn.MultiRNNCell(cells)
        # initial state
        initial_state = rnn.zero_state(X.get_shape()[0], tf.float32)
        # reformat input into a list with elements for each timestep
        inputs = tf.unstack(X, num=X.get_shape()[1], axis=1)
        # connect rnn with input and output
        outputs, state = tf.contrib.rnn.static_rnn(rnn, inputs, initial_state=initial_state)
        # variables of the prediction layer
        initial_weights = tf.random_normal(tf.TensorShape([outputs[-1].get_shape()[1]]), 0.0, 0.1)
        weights = tf.Variable(initial_weights, name='weights')
        bias = tf.Variable(0., dtype=tf.float32, name='bias')
        # logits and prediction
        logits = tf.tensordot(outputs[-1], weights, [[1], [0]]) + bias
        Y_pred = tf.nn.sigmoid(logits, name='probabilities')
        # loss
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                      labels=Y))
        # summaries
        with tf.name_scope('summaries'):
            tf.summary.scalar("loss", loss)
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
        