import tensorflow as tf
from ..models.Model import Model

###################################################

class DNNPlusClassifier(Model):
    
    def __init__(self, model_dir='/tmp/DNNPlusClassifier', 
                 hparams={'learning_rate': 0.01, 'n_hiddens': [30]}):
        Model.__init__(self, model_dir, hparams)
    
    def build_graph(self, data_build_fn=None, data_build_args={}):
        # data and input layer
        data = data_build_fn(**data_build_args)
        X = output = data['X']
        Y = data['Y']
        # we create a list to hold the ops that will be conned to the target
        outputs = [X]
        # get the number of hidden units
        n_hiddens = self.hparams['n_hiddens']
        # get the length of an input vector
        n_input = X.get_shape()[1]
        # iterate over hidden layers
        for ii, n_hidden in enumerate(n_hiddens):
            with tf.name_scope('layer%d' %(ii+1)):
                # create activations based on activation in last layer
                weights_init = tf.random_normal(tf.TensorShape([n_input, n_hidden]), 0.0, 0.1)
                weights = tf.Variable(weights_init, name='weights')
                bias = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32, name='bias')
                # the activation of a layer is created as a linear transformation
                # of the outputs in the last layer
                tdot = tf.tensordot(output, weights, [[1], [0]])
                activation = tf.add(tdot, bias, 'activation')
                # the output of a layer is tanh applied to the activation
                output = tf.nn.tanh(activation, name='output')
                # the output is added to the list of outputs
                outputs.append(output)
                # n_hidden now defines the input shape of the next layer
                n_input = n_hidden
        # we now connect all the output ops to the target
        # first we create a list to hold the logits of each outut
        logits_list = []
        with tf.name_scope('layerTarget'):
            # iterate the output ops to connect them to the target
            for ii, output in enumerate(outputs):
                weights_init = tf.random_normal(tf.TensorShape([output.get_shape()[1]]), 0.0, 0.1)
                weights = tf.Variable(weights_init, name='weights%d' %ii)
                bias = tf.Variable(0., dtype=tf.float32, name='bias%d' %ii)
                tdot = tf.tensordot(output, weights, [[1], [0]])
                logits = tf.add(tdot, bias, 'logits%d' %ii)
                # add the logits of this output layer to logits_list
                logits_list.append(logits)
            # we stack the logits and compute the mean
            logits_stacked = tf.stack(logits_list, axis=1)
            logits_reduced = tf.reduce_mean(logits_stacked, axis=1)
            # the mean of the logits is used for the prediction and loss 
            Y_pred = tf.nn.sigmoid(logits_reduced, name='probabilities')
        # loss
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_reduced, labels=Y),
                              name='loss')
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
        