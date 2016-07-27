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
    global prepared_poems, themes, id_to_word, word_to_id
    print("-"*30)
    print("Data Input")
    print("-"*30)
    #raw_data_IDs, char_to_ix, ix_to_char, vocab_size  = readFile.raw_data('haiku.txt')
    prepared_poems,word_to_id,id_to_word,themes=readPoetry.read_data(os.path.join(config.base_dir,'prepared_poem_data.pkl'))
    prepared_poems=np.random.permutation(prepared_poems)[:config.NUM_POEMS]
    vocab_size=len(id_to_word)
    theme_vocab_size=len(themes)
    print("number of training poems: %d, vocab size: %d, theme size: %d" %(len(prepared_poems),vocab_size,theme_vocab_size))
    args=model.PoetArgs(word_vocab_size=vocab_size,theme_vocab_size=vocab_size,
            learning_rate=config.LEARNING_RATE,init_scale=config.INIT_SCALE,
            num_steps=config.NUM_STEPS,num_layers=config.NUM_LAYERS,
            batch_size=config.BATCH_SIZE,keep_prob=config.KEEP_PROB,
            word_embedding_size=config.WORD_EMBEDDING_SIZE,theme_embedding_size=config.THEME_EMBEDDING_SIZE,
            hidden_size=config.HIDDEN_SIZE)
    # Now save the dictionary, so that we can reload it.
    # we ought to save the training data too, but we don't (to save space)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir,"dict.pkl"),'wb') as f:
        dill.dump({'themes':themes,'word_to_id':word_to_id,'id_to_word':id_to_word,'args':args},f)
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
        p = model.PoetModel(args=args, is_training=True, verbose=True)
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
            epoch_size = ((len(prepared_poems)*len(prepared_poems[0][2]) // args.batch_size) - 1) // args.num_steps
            print("epoch_size: ",epoch_size)
            start_time = time.time()
            costs = 0.0
            iters = 0
            # we backpropogate over a fixed number (num_steps) of GRU units, but we save the final_state
            # so that we can train the rnn to remember things over a much longer string.
            for step, (x, y,cur_themes,new_batch) in enumerate(readPoetry.id_iterator(prepared_poems, args.batch_size,args.num_steps)):
                if new_batch:
                    state = sess.run(p.initial_state,{p.theme_ID: cur_themes})
                summary, cost_on_iter, state, _ = sess.run([p.merged, p.cost, p.final_state, p.train_op],
                                         {p.input_IDs: x,
                                          p.target_IDs: y,
                                          p.initial_state: state,
                                          p.theme_ID:cur_themes})
                costs += cost_on_iter
                iters += args.num_steps

                if step % (epoch_size // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps, epoch est. time rem: %.0f" %
                        (step * 1.0 / epoch_size, np.exp(costs / iters),
                         iters * args.batch_size / (time.time() - start_time), (time.time() - start_time)*epoch_size/(step*1.0)))
                    p.writer.add_summary(summary, i)
            p.writer.flush()
            
            # Save the model every 10 epochs:
            if i>0  and i % 10==0:
                # Save the model in case we want to load it later...
                save_path = saver.save(sess, os.path.join(args.log_dir,"model.ckpt"),global_step=i)
                print("Model saved in file: %s" % save_path)
            
            # Now print a sample:
            sample.sample_from_active_sess(sess,p_sample,word_to_id,id_to_word,themes,args)
        # Save the model one final time.
        # Save the model in case we want to load it later...
        save_path = saver.save(sess, os.path.join(args.log_dir,"model.ckpt"),global_step=i)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main()