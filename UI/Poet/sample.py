#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from . import model
from . import config
import os
import dill

def sample(save_dir=os.path.join(config.base_dir,config.sampling_model_dir),sample_length=32,num_samples=5):
                                            
    with tf.Session() as sess:
        _,p_sample,word_to_id,id_to_word,themes,args=load_model(sess)
        for i in range(num_samples):
            sample_from_active_sess(sess,p_sample,word_to_id,id_to_word,themes,args,sample_length)

def load_model(sess=None,save_dir=os.path.join(config.base_dir,config.sampling_model_dir)):
# Load the saved dictionary
    print("-"*30)
    print("loading the dictionary")
    print("-"*30)
    with open(os.path.join(save_dir,config.sampling_dict),'rb') as f:
        dict_data=dill.load(f)
        id_to_word=dict_data['id_to_word']
        word_to_id=dict_data['word_to_id']
        args=model.PoetArgs()
        args.__dict__=dict_data['args_dict']
        themes=dict_data['themes']
    # Load the saved model
    print("-"*30)
    print("initialization")
    print("-"*30)
    args.num_steps=1
    args.batch_size=1
    args.log_dir=save_dir   
    
    if sess==None:
        sess=tf.Session()
    
    # Initializer
    initializer = tf.random_uniform_initializer(-args.init_scale,args.init_scale)
    with tf.variable_scope("model", reuse = None, initializer = initializer):
        p_sample = model.PoetModel(args=args, is_training = False)
                              
    
    # Initialize the variables
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()


    print("-"*30)
    print("loading the model")
    print("-"*30)
    saver.restore(sess, os.path.join(save_dir,config.sampling_model))
    return sess,p_sample,word_to_id,id_to_word,themes,args
    

def sample_from_active_sess(sess,p_sample,word_to_id,id_to_word,themes,args,sample_length=32):
    print("-"*30)
    print("Sampling")
    print("-"*30)
    
    #Choose a theme:
    sample_theme=np.random.choice(list(themes.keys()))
    sample_theme_ID=word_to_id[sample_theme]
    
    print("Theme: "+sample_theme+"\n")
    txt=sample_from_active_sess_with_theme(sess,p_sample,word_to_id,id_to_word,sample_theme,args,sample_length)
    print('----\n %s \n----' % (txt, ))
    with open(os.path.join(args.log_dir,"output.txt"),'a') as f_out:
        f_out.write("Theme: "+id_to_word[sample_theme_ID]+"\n")
        f_out.write('----\n %s \n----' % (txt.replace('<sop>','').replace('<eop>',''), ))

def choose_theme(theme_possibilities,specified_theme_dict):
    rem_themes,weights = clean_themes(theme_possibilities,specified_theme_dict)
    return np.random.choice(rem_themes,p=weights)
    
def clean_themes(theme_possibilities,specified_theme_dict):
    # Remove unwanted themes
    processed_specified_theme_dict={}
    for thm in list(specified_theme_dict.keys()):
        thm_parts=thm.lower().split(" ")
        max_cnt = 0
        max_part = ''
        for thm_part in thm_parts:
            if thm_part in theme_possibilities and theme_possibilities[thm_part] > max_cnt:
                max_cnt=theme_possibilities[thm_part]
                max_part=thm_part
                #del(specified_theme_dict[thm])
        if max_cnt > 0 :
            processed_specified_theme_dict[max_part]=specified_theme_dict[thm]
    rem_themes = list(processed_specified_theme_dict.keys())
    if len(rem_themes) > 0:
        # Sum up the weights and renormalize
        rel_weights = []
        for thm in rem_themes:
            rel_weights+=[processed_specified_theme_dict[thm]*1.]
        weights=np.array(rel_weights)/sum(rel_weights)
    else:
        rem_themes=['<unk>']
        weights=[1.]
    return rem_themes,weights
        

def sample_from_active_sess_with_theme(sess,p_sample,word_to_id,id_to_word,sample_theme,args,sample_length=32):
    sample_theme_ID=word_to_id[sample_theme]
    state = sess.run(p_sample.initial_state,{p_sample.theme_ID: [sample_theme_ID]})
    # generate the sample:
    ix=word_to_id['<sop>']
    txt=''
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
        assert len(p_out_prob)==args.word_vocab_size, "There is a size mismatch between the softmax output size: %d, and the vocab size: %d" %(len(p_out_prob),args.word_vocab_size)
        ix=sample
        txt+=id_to_word[ix]+" "
    return txt
    
    
def multi_theme_sample(sess,p_sample,word_to_id,id_to_word,sample_themes,sample_theme_weights,args,sample_length=32):
    sample_theme_weights=np.array([sample_theme_weights])
    sample_themes_IDs=[word_to_id[sample_theme] for sample_theme in sample_themes]
    states = [sess.run(p_sample.initial_state,{p_sample.theme_ID: [sample_theme_ID]}) for sample_theme_ID in sample_themes_IDs]
    # generate the sample:
    ix=word_to_id['<sop>']
    txt=''
    for i in range(config.SAMPLE_LENGTH):
        out_prob_states = [sess.run([p_sample.output_prob,p_sample.final_state],
                                 {p_sample.input_IDs: [[ix]],
                                  p_sample.initial_state: state}) for state in states]
        states=[o_p_s[1] for o_p_s in out_prob_states]
        out_prob=sample_theme_weights.dot([o_p_s[0].reshape(-1) for o_p_s in out_prob_states])
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
        assert len(p_out_prob)==args.word_vocab_size, "There is a size mismatch between the softmax output size: %d, and the vocab size: %d" %(len(p_out_prob),args.word_vocab_size)
        ix=sample
        txt+=id_to_word[ix]+" "
    return clean_output(txt)
    
def clean_output(txt):
    return txt.replace(' <eop>','')\
        .replace(' i ',' I ')\
        .replace(" \' ","\'")\
        .replace(" ,",",")\
        .replace(" .",".")\
        .replace(" ?","?")\
        .replace(" !","!")\
        .replace('\n','<br>')

if __name__ == '__main__':
    sample()