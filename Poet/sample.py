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
        word_to_id=dict_data['word_to_id']
        args=dict_data['args']
        themes=dict_data['themes']
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
        sample_from_active_sess(sess,p_sample,word_to_id,id_to_word,themes,args,sample_length)
        

def sample_from_active_sess(sess,p_sample,word_to_id,id_to_word,themes,args,sample_length=32):
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
    
    #Choose a theme:
    sample_theme=np.random.choice(list(themes.keys()))
    sample_theme_ID=word_to_id[sample_theme]
    
    print("Theme: "+id_to_word[sample_theme_ID]+"\n")
    state = sess.run(p_sample.initial_state,{p_sample.theme_ID: [sample_theme_ID]})
    # generate the sample:
    ix=word_to_id['<sop>']
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
        # now choose a random number
        rn = np.random.rand(1)[0]
        cnt=0.0
        p_out_prob=out_prob.ravel()
        for i in range(len(p_out_prob)):
            cnt+=p_out_prob[i]
            if rn <= cnt:
                sample=i
                break
        ix=range(args.word_vocab_size)[sample]
        txt+=id_to_word[ix]+" "
    print('----\n %s \n----' % (txt, ))
    with open(os.path.join(args.log_dir,"output.txt"),'a') as f_out:
        f_out.write("Theme: "+id_to_word[sample_theme_ID]+"\n")
        f_out.write('----\n %s \n----' % (txt.replace('<sop>','').replace('<eop>',''), ))


if __name__ == '__main__':
    sample()