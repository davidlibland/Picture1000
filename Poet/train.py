#!/usr/bin/env python3

import time
import tensorflow as tf
import numpy as np
import model
import readFile
import copy

max_epochs=10
sample_length=200



def main():
    global raw_data_IDs, char_to_ix, ix_to_char
    print("-"*30)
    print("Data Input")
    print("-"*30)
    raw_data_IDs, char_to_ix, ix_to_char, vocab_size  = readFile.raw_data('haiku.txt')
    args=model.PoetArgs(word_vocab_size=vocab_size,theme_vocab_size=vocab_size,num_steps=32,batch_size=1,keep_prob=1)
    train_rnn(args)

def train_rnn(args,text_file=None,restore=False):
    #if restore:
    #    load_rnn()
    print("-"*30)
    print("initialization")
    print("-"*30)
    sample_args=copy.copy(args)
    sample_args.num_steps=1
    sample_args.batch_size=1
    
    # Initializer
    initializer = tf.random_uniform_initializer(-args.init_scale,args.init_scale)
    with tf.variable_scope("model", reuse = None, initializer = initializer):
        p = model.PoetModel(args=args, is_training=True)
    #with tf.variable_scope("model", reuse = True, initializer = initializer):
        tf.get_variable_scope().reuse_variables()
        p_sample = model.PoetModel(args=sample_args, is_training = False)
                                            
    with tf.Session() as sess:
        print("-"*30)
        print("training")
        print("-"*30)
        
        # Initialize the variables
        tf.initialize_all_variables().run()
        
        for i in range(max_epochs):
            epoch_size = ((len(raw_data_IDs) // args.batch_size) - 1) // args.num_steps
            print("epoch_size: ",epoch_size)
            start_time = time.time()
            costs = 0.0
            iters = 0
            state = sess.run(p.initial_state,{p.theme_ID: [0]})
            # we backpropogate over a fixed number (num_steps) of GRU units, but we save the final_state
            # so that we can train the rnn to remember things over a much longer string.
            for step, (x, y) in enumerate(readFile.id_iterator(raw_data_IDs, args.batch_size,args.num_steps)):
                summary, cost_on_iter, state, _ = sess.run([p.merged, p.cost, p.final_state, p.train_op],
                                         {p.input_IDs: x,
                                          p.target_IDs: y,
                                          p.initial_state: state, p.theme_ID:[0]})
                costs += cost_on_iter
                iters += args.num_steps

                if step % (epoch_size // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps" %
                        (step * 1.0 / epoch_size, np.exp(costs / iters),
                         iters * args.batch_size / (time.time() - start_time)))
                    p.writer.add_summary(summary, i)
            p.writer.flush()
            
            #with tf.Session(graph=p_sample.graph) as sess_sample:
            # Initialize the variables
            #tf.initialize_all_variables().run()
            state = sess.run(p_sample.initial_state,{p_sample.theme_ID: [0]})
            # Print a sample:
            ix=np.random.choice(range(args.word_vocab_size))
            txt=''
            # we generate the sample by applying the GRU recursively to it's ouput
            for i in range(sample_length):
                out_prob,state = sess.run([p_sample.output_prob,p_sample.final_state],
                                         {p_sample.input_IDs: [[ix]],
                                          p_sample.initial_state: state})
                # we save the GRU state so we can recursively apply it.
                # we also get a probability of what possible outputs the GRU predicts at this step
                # make a weighted choice of output with those probabilities, this will be the sampled
                # output at this step, as well as the input conditioning the GRU's output for the next step.
                ix = np.random.choice(range(args.word_vocab_size), p=out_prob.ravel())
                txt+=ix_to_char[ix]
            #txt = sample_rnn()
            print('----\n %s \n----' % (txt, ))
        # Save the model in case we want to load it later...
    #save_rnn()


if __name__ == '__main__':
    main()