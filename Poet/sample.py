#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from . import model
from . import config
import os
import dill

def sample(save_dir=os.path.join(config.base_dir,'Saved_Model_Dir'),sample_length=32):
    # Load the saved dictionary
    print("-"*30)
    print("loading the dictionary")
    print("-"*30)
    with open(os.path.join(save_dir,"dict.pkl"),'rb') as f:
        dict_data=dill.load(f)
        id_to_word=dict_data['id_to_word']
        args=dict_data['args']
        weights=dict_data['weights']
    # Load the saved model
    print("-"*30)
    print("initialization")
    print("-"*30)
    args.num_steps=1
    args.batch_size=1
    
    # Initializer
    initializer = tf.random_uniform_initializer(-args.init_scale,args.init_scale)
    with tf.variable_scope("model", reuse = None, initializer = initializer):
        p_sample = model.PoetModel(args=args, is_training = False)
    
                                            
    with tf.Session() as sess:
        print("-"*30)
        print("training")
        print("-"*30)
        
        # Initialize the variables
        tf.initialize_all_variables().run()
        
        saver = tf.train.Saver()
        
        
        print("-"*30)
        print("loading the model")
        print("-"*30)
        ckpt = tf.train.get_checkpoint_state(args.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored.")
        else:
            print("No checkpoint file found")
        
        sample_theme_ID=-1
        while sample_theme_ID not in weights or id_to_word[sample_theme_ID] == '<eos>' :
            sample_theme_ID=np.random.choice(list(weights))
        print("Theme: "+id_to_word[sample_theme_ID]+"\n")
        state = sess.run(p_sample.initial_state,{p_sample.theme_ID: [sample_theme_ID]})
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
        

if __name__ == '__main__':
    sample()