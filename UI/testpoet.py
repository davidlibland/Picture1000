#!/usr/bin/env python3

import tensorflow as tf
import Poet.sample as s
with tf.Session() as sess:
	_,p_sample,word_to_id,id_to_word,themes,args=s.load_model(sess)
	th,wei = s.clean_themes(themes,{'flower':0.5,'canyon':0.5})
	for _ in range(10):
	    txt = s.multi_theme_sample(sess,p_sample,word_to_id,id_to_word,th,wei,args)
	    print("-"*30+"\n"+txt)