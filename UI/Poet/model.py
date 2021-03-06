#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
import datetime
import os
from . import config
#from tensorflow.models.rnn_* import rnn
#import numpy as np

class PoetArgs(object):
    def __init__(self,batch_size = 4, num_steps = 128, num_layers = 2, learning_rate = 0.01, max_grad_norm = 5.,
                    init_scale = 0.05, hidden_size = 128, keep_prob = .5, word_embedding_size = 32,
                    word_vocab_size = 128, theme_embedding_size = 32, theme_vocab_size = 128,
                    log_dir=os.path.join(config.base_dir,"tmp/PoetLog"+datetime.datetime.now().isoformat())):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.init_scale = init_scale
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.word_embedding_size = word_embedding_size
        self.theme_embedding_size = theme_embedding_size
        self.word_vocab_size = word_vocab_size
        self.theme_vocab_size = theme_vocab_size
        self.log_dir=log_dir
    
    def __repr__(self):
        return (str(self.__dict__))

class FastGRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.nn.softsign):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(1, 2, tf.nn.rnn_cell._linear([inputs, state],
                                             2 * self._num_units, True, 1.0))
        # We use (a shifted & rescaled) softsign rather than a sigmoid
        r, u = 0.5*(1+tf.nn.softsign(r)), 0.5*(1+tf.nn.softsign(u))
      with vs.variable_scope("Candidate"):
        c = self._activation(tf.nn.rnn_cell._linear([inputs, r * state],
                                     self._num_units, True))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

class PoetModel(object):
    def __init__(self,args, is_training=False, verbose=False):
        # First we store the configuration object as a component.
        self._args = args
        
        # Create some placeholders for the input/output data
        self._input_IDs = tf.placeholder(tf.int32, [self.args.batch_size, self.args.num_steps],name="input_IDs") # batch_size x num_steps
        self._target_IDs = tf.placeholder(tf.int32, [self.args.batch_size, self.args.num_steps],name="target_IDs") # batch_size x num_steps
        # Create a placeholder for the theme
        self._theme_ID = tf.placeholder(tf.int32,[self.args.batch_size],name="theme_ID")
        # Create a placeholder for the `strength' of the theme
        # the `strength' is inversely proportional to how common the theme is between the given poems.
        # For instance, the word `the' is extremely common, so it not a strong theme of any given poem,
        # While the word 'disgust' is far less common, and hence a stronger `theme' if found in any poem.
        # Note: this placeholder is only used during training.
        #self._theme_strength=tf.placeholder(tf.float32,[self.args.batch_size],name="theme_strength")
        #theme_strength_reshaped=tf.reshape(self.theme_strength,[-1,1]) 
                
        if is_training:
            name_scope = "rnnPoetTrainer"
        else:
            name_scope = "rnnPoet"
        with tf.name_scope(name_scope):
            
            # We now create learnable embeddings for the words and the themes:
            with tf.name_scope('embedding'):
                # for now, embedding needs to be placed on the cpu.
                with tf.device("/cpu:0"):
                    # word_embedding = tf.get_variable("word_embedding", [self.args.word_vocab_size, self.args.word_embedding_size])
                    word_embedding = tf.get_variable("word_embedding",
                                        [self.args.word_vocab_size, self.args.hidden_size],
                                        initializer=tf.random_normal_initializer(0, 1/self.args.hidden_size))
                    inputs = tf.nn.embedding_lookup(word_embedding, self.input_IDs,name="word_vect")
                    # theme_embedding = tf.get_variable("theme_embedding", [self.args.theme_vocab_size, self.args.theme_embedding_size])
                    theme_embedding = tf.get_variable("theme_embedding",
                                        [self.args.theme_vocab_size, self.args.hidden_size],
                                        initializer=tf.random_normal_initializer(0, 1/self.args.hidden_size))
                    theme = tf.nn.embedding_lookup(theme_embedding, self.theme_ID,name="theme_vect")
                if is_training and self.args.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, self.args.keep_prob)
                    theme = tf.nn.dropout(theme, self.args.keep_prob)
            
            
            # Create the GRU Cell
            size=self.args.hidden_size #+self.args.theme_embedding_size
            cell=FastGRUCell(size,activation=tf.nn.softsign)
            
            if is_training and self.args.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                            cell, output_keep_prob=self.args.keep_prob)
            # Now stack the GRU Cell on top of itself self.args.num_layers times
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.args.num_layers)
            
            
            # The initial state should be the concatination of the theme and the zero state.
            # dim 0 is batch size, so we concatinate along dim 1.
            # next, we stack this initial state on top of itself self.args.num_layers times along dim 1;
            # this way, each layer is presented with the theme.
            # self._initial_state = tf.tile(tf.concat(1,[tf.zeros([self.args.batch_size, self.args.hidden_size], tf.float32),theme]),
            #                                [1,self.args.num_layers],name="initial_state") 
            self._initial_state = tf.tile(tf.nn.softsign(theme),
                                            [1,self.args.num_layers],name="initial_state") 
            # self._initial_state = cell.zero_state(self.args.batch_size,tf.float32)

            # Link the GRU cell sequentially to itself:
           #  outputs = []
#                 state=self.initial_state
#                 # Make sure we save the scope as rnn_scope so that we can reuse the trained GRU weights later when sampling.
#                 with tf.variable_scope("RNN"):
#                     for time_step in range(self.args.num_steps):
#                         # After the first unit, we want to reuse the GRU weights
#                         if time_step > 0:
#                             tf.get_variable_scope().reuse_variables()
#                         (cell_output, state) = cell(inputs[:, time_step, :], state)
#                         outputs.append(cell_output)
                    
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, self.args.num_steps, inputs)]
            outputs, state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)

            self._final_state=state #Save the final state so we can access it later when training.
            
            
            # Transform the GRU output to logits via a learnt linear transform
            output = tf.reshape(tf.concat(1, outputs), [-1, size])
            softmax_w = tf.get_variable("softmax_w", [size, self.args.word_vocab_size],initializer=tf.random_normal_initializer(0,1/size))
            softmax_b = tf.get_variable("softmax_b", [self.args.word_vocab_size],initializer=tf.constant_initializer(0.))
            logits = tf.matmul(output, softmax_w) + softmax_b
            self._output_prob=tf.nn.softmax(logits)
            
            
            # Compute the loss
            with tf.name_scope('total_loss'):
                #weight_list = [theme_strength_reshaped for _ in range(self.args.num_steps)]
                #weights=tf.reshape(tf.concat(1, weight_list), [-1])
                loss = tf.nn.seq2seq.sequence_loss_by_example(
                    [logits],
                    [tf.reshape(self.target_IDs, [-1])],
                    [tf.ones([self.args.batch_size * self.args.num_steps])])
                self._cost = tf.reduce_sum(loss) / self.args.batch_size
            if verbose:
                tf.histogram_summary("output weights", softmax_w)
                tf.histogram_summary("output biases", softmax_b)
                tf.histogram_summary("word embedding", word_embedding)
                tf.histogram_summary("theme embedding", theme_embedding)
                tf.scalar_summary("output weights zero fraction", tf.nn.zero_fraction(softmax_w))
                tf.scalar_summary("output biases zero fraction", tf.nn.zero_fraction(softmax_b))
                tf.scalar_summary("word embedding zero fraction", tf.nn.zero_fraction(word_embedding))
                tf.scalar_summary("theme embedding zero fraction", tf.nn.zero_fraction(theme_embedding))
                tf.scalar_summary('cross entropy', self.cost)
                # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
                self._merged = tf.merge_all_summaries()
                self._writer = tf.train.SummaryWriter(self.args.log_dir,
                                                        tf.get_default_graph())

            if not is_training:
                return
                
            with tf.name_scope('train'):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),self.args.max_grad_norm)
                optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
                self._train_op = optimizer.apply_gradients(zip(grads, tvars))
                    
                
                    
    @property
    def args(self):
        return self._args
    
    @property
    def input_IDs(self):
        return self._input_IDs
    
    @property
    def target_IDs(self):
        return self._target_IDs
    
    @property
    def theme_ID(self):
        return self._theme_ID
    
#     @property
#     def theme_strength(self):
#         return self._theme_strength
    
    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def cost(self):
        return self._cost
    
    @property
    def final_state(self):
        return self._final_state
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def output_prob(self):
        return self._output_prob
    
    @property
    def merged(self):
        return self._merged
    
    @property
    def writer(self):
        return self._writer

#PoetModel(PoetArgs(),is_training=True)