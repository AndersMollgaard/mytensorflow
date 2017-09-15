import tensorflow as tf
import os
import pickle

############### Helper functions ##################

def metrics_saver(save_dict, model_dir):
    try:
        with open('%s/metrics.p' %model_dir, 'rb') as f:
            save_list = pickle.load(f)
    except:
        save_list = []
    save_list.append(save_dict)
    with open('%s/metrics.p' %model_dir, 'wb') as f:
        pickle.dump(save_list, f)

############### Model parent class #################

class Model():
    
    def __init__(self, model_dir='/tmp/tmpmodel', hparams={'learning_rate': 0.01}):
        self.model_dir = model_dir
        self.hparams = hparams
        # create a directory string to a hparams subfolder (folder created during run)
        if hparams:
            # create a string with the parameter values
            hparams_string = '#'.join([ '='.join([key, str(val)]) for key, val in 
                                        sorted(hparams.items(), key=lambda x: x[0])])
            # remove chars that do not work in file names
            hparams_string = ''.join([char for char in hparams_string 
                                      if char not in ['[', ']', '(', '(']])
            self.model_dir_hparams = '%s/%s' %(self.model_dir, hparams_string)
        else:
            self.model_dir_hparams = self.model_dir
        # create model directory if it does not exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def predict(self, data_build_fn=None, data_build_args={}):
        # instantiate a graph and a session
        with tf.Graph().as_default(), tf.Session() as sess:
            print('Predicting.')
            # build graph
            self.build_graph(data_build_fn, data_build_args)
            # restore variables
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_dir_hparams)
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
            except:
                print('Failed to restore checkpoint from %s' 
                      %ckpt.model_checkpoint_path)
            Y_pred = self.sess.run(self.Y_pred)
        return Y_pred
    
    def train(self, data_build_fn=None, data_build_args={}, steps=1, steps_log=100, 
              steps_save=1000):
        # instantiate a graph and a session
        with tf.Graph().as_default(), tf.Session() as sess:
            print('Training.')
            # build graph
            self.build_graph(data_build_fn, data_build_args)
            # initialize a writer for summaries
            writer = tf.summary.FileWriter(self.model_dir_hparams, 
                                           graph=tf.get_default_graph())
            # initialize a saver for checkpoints
            saver = tf.train.Saver()
            # restore last checkpoint if it exists
            ckpt = tf.train.get_checkpoint_state(self.model_dir_hparams)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # loss_sum is used to calc a moving average
            loss_sum = 0
            for step in range(1, steps+1):
                # make an update
                ops_run = [self.optimizer, self.loss, self.global_step]
                _, loss, global_step = sess.run(ops_run)
                # add loss to moving average
                loss_sum += loss
                # every steps_print
                if step % steps_log == 0:
                    # print moving average
                    loss_avg = loss_sum / steps_log
                    print('Local loss at step %d: %s' %(global_step, loss_avg))
                    loss_sum = 0.
                    # write summary
                    summary = sess.run(self.summary)
                    writer.add_summary(summary, global_step=global_step)
                # every steps_save
                if step % steps_save == 0:
                    saver.save(sess, os.path.join(self.model_dir_hparams, "model.ckpt"),
                        global_step=self.global_step)
            # close the writer
            writer.close()
        
    def evaluate(self, data_build_fn, data_build_args):
        # instantiate a graph and a session
        with tf.Graph().as_default(), tf.Session() as sess:
            print('Evaluating.')
            # build graph
            self.build_graph(data_build_fn, data_build_args)
            # restore variables
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_dir_hparams)
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
            except:
                print('Failed to restore checkpoint from %s' 
                      %ckpt.model_checkpoint_path)
            # get metric ops
            metrics = list(self.eval_metric_ops.keys())
            metric_ops = list(self.eval_metric_ops.values())
            # evaluate
            while True:
                try:
                    metric_ops_eval = sess.run(metric_ops)
                # catch OutOfRangeError to handle end of file reading
                # catch InvalidArgumentError to catch invalid input to reshape for 
                # LSTM feeding, which probably occurs for the last batch. Think
                # of a better solution 
                except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError) as e:
                    break
            # contain evaluated metrics in dict and print
            metrics_final = {metric: val for metric, val in zip(metrics, metric_ops_eval)}
            self.metrics_final = metrics_final
            print(metrics_final)
            # create a save dictionary with metrics and parameters
            save_dict = {'global_step': self.global_step.eval()}
            save_dict.update(self.hparams)
            save_dict.update(metrics_final)
            # append save_dict to the list in the file "metrics.p"
            metrics_saver(save_dict, self.model_dir)