#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import readFile
import time
from six.moves import cPickle

batch_size=1
#seq_length=128
num_steps=128
learning_rate=0.01
max_grad_norm=5.
init_scale = 0.05

hidden_size=128
max_epochs=32

sample_length=200

print("-"*30)
print("Data Input")
print("-"*30)

raw_data_IDs, char_to_ix, ix_to_char, vocab_size  = readFile.raw_data('haiku.txt')

print("-"*30)
print("initialization")
print("-"*30)

sess=tf.InteractiveSession()


# Create a list of input/target placeholders
input_IDs = tf.placeholder(tf.int32, [batch_size, num_steps]) # batch_size x num_steps
target_IDs = tf.placeholder(tf.int32, [batch_size, num_steps]) # batch_size x num_steps
sample_input_ID=tf.placeholder(tf.int32,[1])


# convert inputs to 1-hot-labeling
embedding=tf.diag([1.]*vocab_size, name="embedding")
inputs = tf.gather(embedding,input_IDs,name="input")
sample_input=tf.gather(embedding,sample_input_ID,name="sample_input")

# Initializer
initializer = tf.random_uniform_initializer(-init_scale,init_scale)
# Define the model
with tf.variable_scope("model", reuse=None, initializer=initializer):
    # Create the GRU Cell
    cell=tf.nn.rnn_cell.GRUCell(hidden_size)
    initial_state = cell.zero_state(batch_size, tf.float32)

    # Link the GRU cell sequentially to itself:
    outputs=[]
    state=initial_state
    # Make sure we save the scope as rnn_scope so that we can reuse the trained GRU weights later when sampling.
    with tf.variable_scope("RNN") as rnn_scope:
        for time_step in range(num_steps):
            # After the first unit, we want to reuse the GRU weights
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
    final_state=state #Save the final state so we can access it later when training.

    # Transform the GRU output to logits via a learnt linear transform
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b

with tf.variable_scope(rnn_scope, reuse=True):
    # Create a sampling GRU Cell
    #sample_cell=tf.nn.rnn_cell.GRUCell(hidden_size)
    sample_input_state=cell.zero_state(1, tf.float32)
    sample_output,sample_output_state=cell(sample_input,sample_input_state)
sample_output_prob=tf.nn.softmax(tf.matmul(tf.reshape(tf.concat(1,sample_output), [-1, hidden_size]), softmax_w) + softmax_b)
    

# Compute the loss
with tf.name_scope('total'):
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(target_IDs, [-1])],
        [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss) / batch_size
tf.scalar_summary('cross entropy', cost)


with tf.name_scope('train'):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.apply_gradients(zip(grads, tvars))
        
        
 # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./tmp/HaikuTrainSum',
                                        sess.graph)


# Initialize the variables...
tf.initialize_all_variables().run()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(tf.trainable_variables())

for var in tf.trainable_variables():
    print(var.name)

def train_rnn(text_file=None,restore=False):
    if restore:
        load_rnn()
    print("-"*30)
    print("training")
    print("-"*30)
    for i in range(max_epochs):
        epoch_size = ((len(raw_data_IDs) // batch_size) - 1) // num_steps
        print("epoch_size: ",epoch_size)
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = initial_state.eval()
        # we backpropogate over a fixed number (num_steps) of GRU units, but we save the final_state
        # so that we can train the rnn to remember things over a much longer string.
        for step, (x, y) in enumerate(readFile.id_iterator(raw_data_IDs, batch_size,num_steps)):
            summary, cost_on_iter, state, _ = sess.run([merged, cost, final_state, train_step],
                                     {input_IDs: x,
                                      target_IDs: y,
                                      initial_state: state})
            costs += cost_on_iter
            iters += num_steps

            if step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / epoch_size, np.exp(costs / iters),
                     iters * batch_size / (time.time() - start_time)))
                train_writer.add_summary(summary, i)
                
        
        # Print a sample:
        txt = sample_rnn()
        print('----\n %s \n----' % (txt, ))
    # Save the model in case we want to load it later...
    save_rnn()
        
# By default we do not restore the model each time we sample it... Usually it is better to load it 
# once, and then sample it multiple times...
def sample_rnn(seed=None,restore=False):
    if restore:
        load_rnn()
    state = initial_state.eval()
    # If we're given a seed, then we want to initialize the model with that seed:
    if seed != None:
        # First we translate the seed (in characters) to numbers using our dictionary
        seed_IDs = [char_to_ix[ch] for ch in seed]
        # Then we feed that list of numbers recursively into the rnn
        for seed_ix in seed_IDs[:-2]:
            out_prob,state = sess.run([sample_output_prob,sample_output_state],
                                     {sample_input_ID: [seed_ix],
                                      sample_input_state: state})
        # Finally, we record the last number in our seed as ix - which we use as the next input of the rnn
        ix=seed_IDs[-1]
        txt=seed
    else:
        ix=np.random.choice(range(vocab_size))
        txt=''
    # we generate the sample by applying the GRU recursively to it's ouput
    for i in range(sample_length):
        out_prob,state = sess.run([sample_output_prob,sample_output_state],
                                 {sample_input_ID: [ix],
                                  sample_input_state: state})
        # we save the GRU state so we can recursively apply it.
        # we also get a probability of what possible outputs the GRU predicts at this step
        # make a weighted choice of output with those probabilities, this will be the sampled
        # output at this step, as well as the input conditioning the GRU's output for the next step.
        ix = np.random.choice(range(vocab_size), p=out_prob.ravel())
        txt+=ix_to_char[ix]
    return txt

def save_rnn():
    # Save the vocabulary
    with open('./tmp/dict.pkl', 'wb') as f:
        cPickle.dump((char_to_ix, ix_to_char), f)
    # Save the weights.
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

#
def load_rnn():
    print("-"*30)
    print("loading model")
    print("-"*30)
    # Restore variables from disk.
    ckpt = tf.train.get_checkpoint_state("./tmp/")
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored.")
    else:
        print("No checkpoint file found")
    # Restore the dictionary. 
    with open('./tmp/dict.pkl','rb') as f:
        global char_to_ix, ix_to_char
        char_to_ix, ix_to_char = cPickle.load(f)
        print("Dictionary restored")
    #saver.restore(sess, "./tmp/model.ckpt")
    
    
def main():
    #train_rnn()
    print(sample_rnn(seed='love',restore=True))
    print("-"*30)
    print(sample_rnn(seed='The'))
    print("-"*30)
    print(sample_rnn(seed='my'))
    print("-"*30)
    print(sample_rnn(seed='in'))
    print("-"*30)
    print(sample_rnn(seed='wild beaches'))
    print("-"*30)
    print(sample_rnn(seed='the bottle'))
    print("-"*30)
    print(sample_rnn(seed='at opposite'))
if __name__ == "__main__":
    main()