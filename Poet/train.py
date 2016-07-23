#!/usr/bin/env python3

import time
import tensorflow as tf
import numpy as np
from . import model
from . import readPoetry
import copy
import os
from . import sample
from . import config
import dill



def main():
    global poems_encoded, weights, id_to_word
    print("-"*30)
    print("Data Input")
    print("-"*30)
    #raw_data_IDs, char_to_ix, ix_to_char, vocab_size  = readFile.raw_data('haiku.txt')
    poems_encoded,weights,word_to_id,id_to_word=readPoetry.read_data(os.path.join(config.base_dir,'database.pkl'))
    poems_encoded=np.random.permutation(poems_encoded)[:1024]
    vocab_size=len(id_to_word)
    args=model.PoetArgs(word_vocab_size=vocab_size,theme_vocab_size=vocab_size,num_steps=32,batch_size=16,keep_prob=0.5,hidden_size=512)
    # Now save the dictionary, so that we can reload it.
    # we ought to save the training data too, but we don't (to save space)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir,"dict.pkl"),'wb') as f:
        dill.dump({'weights':weights,'word_to_id':word_to_id,'id_to_word':id_to_word,'args':args},f)
    train_rnn(args)
    tf.reset_default_graph() 
    sample.sample(args.log_dir)

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
        
        saver = tf.train.Saver()
        for i in range(config.MAX_EPOCHS):
            epoch_size = ((len(poems_encoded)*len(poems_encoded[0]) // args.batch_size) - 1) // args.num_steps
            print("epoch_size: ",epoch_size)
            start_time = time.time()
            costs = 0.0
            iters = 0
            #state = sess.run(p.initial_state,{p.theme_ID: [0]})
            # we backpropogate over a fixed number (num_steps) of GRU units, but we save the final_state
            # so that we can train the rnn to remember things over a much longer string.
            for step, (x, y,themes,strengths,new_batch) in enumerate(readPoetry.id_iterator(poems_encoded,weights, args.batch_size,args.num_steps)):
                if new_batch:
                    state = sess.run(p.initial_state,{p.theme_ID: themes})
                summary, cost_on_iter, state, _ = sess.run([p.merged, p.cost, p.final_state, p.train_op],
                                         {p.input_IDs: x,
                                          p.target_IDs: y,
                                          p.initial_state: state,
                                          p.theme_ID:themes,
                                          p.theme_strength:strengths})
                costs += cost_on_iter
                iters += args.num_steps

                if step % (epoch_size // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps, epoch est. time rem: %.0f" %
                        (step * 1.0 / epoch_size, np.exp(costs / iters),
                         iters * args.batch_size / (time.time() - start_time), (time.time() - start_time)*epoch_size/(step*1.0)))
                    p.writer.add_summary(summary, i)
            p.writer.flush()
            
            #with tf.Session(graph=p_sample.graph) as sess_sample:
            # Initialize the variables
            #tf.initialize_all_variables().run()
            sample_theme_ID=-1
            while sample_theme_ID not in weights or id_to_word[sample_theme_ID] == '<eos>' :
                sample_theme_ID=np.random.choice(np.array(poems_encoded).reshape([-1]))
            print("Theme: "+id_to_word[sample_theme_ID]+"\n")
            state = sess.run(p_sample.initial_state,{p_sample.theme_ID: [sample_theme_ID]})
            # Print a sample:
            ix=np.random.choice(range(args.word_vocab_size))
            txt=''
            # we generate the sample by applying the GRU recursively to it's ouput
            for i in range(config.SAMPLE_LENGTH):
                out_prob,state = sess.run([p_sample.output_prob,p_sample.final_state],
                                         {p_sample.input_IDs: [[ix]],
                                          p_sample.initial_state: state})
                # we save the GRU state so we can recursively apply it.
                # we also get a probability of what possible outputs the GRU predicts at this step
                # make a weighted choice of output with those probabilities, this will be the sampled
                # output at this step, as well as the input conditioning the GRU's output for the next step.
                
                
                # we want to choose the next word as follows:
                # ix = np.random.choice(range(args.word_vocab_size), p=out_prob.ravel())
                # but for a large number of words, the probabilities may not sum close enough to 1,
                # so we do the following instead:
                #now choose a random number
                if True or id_to_word[ix]!='<eos>':
                    rn = np.random.rand(1)[0]
                    cnt=0.0
                    p_out_prob=out_prob.ravel()
                    for i in range(len(p_out_prob)):
                        cnt+=p_out_prob[i]
                        if rn <= cnt:
                            sample=i
                            break
                    ix=range(args.word_vocab_size)[sample]
                else:
                    for i in range(10):
                        rn = np.random.rand(1)[0]
                        cnt=0.0
                        p_out_prob=out_prob.ravel()
                        for i in range(len(p_out_prob)):
                            cnt+=p_out_prob[i]
                            if rn <= cnt:
                                sample=i
                                break
                        ix=range(args.word_vocab_size)[sample]
                        if id_to_word[ix]!='<eos>':
                            break
                txt+=id_to_word[ix]+" "
            #txt = sample_rnn()
            print('----\n %s \n----' % (txt.replace('<eos>',"\n"), ))
            with open(os.path.join(args.log_dir,"output.txt"),'a') as f_out:
                f_out.write("Theme: "+id_to_word[sample_theme_ID]+"\n")
                f_out.write('----\n %s \n----' % (txt.replace('<eos>',"\n"), ))
        # Save the model in case we want to load it later...
        save_path = saver.save(sess, os.path.join(args.log_dir,"model.ckpt"))
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main()