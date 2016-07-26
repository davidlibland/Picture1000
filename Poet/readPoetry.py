#!/usr/bin/env python3

# This is a modified version of 
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
# which had the following license:
# ==============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections import Counter
import numpy as np
import dill
    
def read_data(data_path):
    with open(data_path,'rb') as f:
        poems_encoded,words,word_to_id,id_to_word = dill.load(f)
        common_words={}
        for w in words:
            if words[w] > 1:
                common_words[w]=words[w]
        weights={}
        for w in common_words:
            weights[word_to_id[w]]=1/words[w]
        return poems_encoded,weights,word_to_id,id_to_word

def id_iterator(raw_data, weights, batch_size, num_steps):
    """Iterate on the raw data, a list of poems (with words converted to ids)
    This generates batch_size pointers into the raw text data, and allows
    minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls, should be a divisor of poem_length+1.
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    
    #shuffled_data=np.array(raw_data,dtype=np.int32)    
    shuffled_data=np.array(np.random.permutation(raw_data),dtype=np.int32)

    data_len,poem_length = shuffled_data.shape
    batch_len = data_len // batch_size

    for j in range(batch_len):
        # Load the current batch
        current_batch=shuffled_data[j:j+batch_size,:]
        # Randomly pick themes from the allowed themes
        themes=[]
        strengths=[]
        for i in range(batch_size):
            themes+=[None]
            while themes[i] not in weights:
                themes[i]=np.random.choice(current_batch[i,:])
            strengths.append(weights[themes[i]])
        for i in range(poem_length//num_steps):
            x=current_batch[:,i*num_steps:(i+1)*num_steps]
            y = current_batch[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x,y,themes,strengths,i==0)