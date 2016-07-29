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
        data = dill.load(f)
        return data['prepared_poems'],data['word_to_id'],data['id_to_word'],data['themes']

def id_iterator(prepared_poems, batch_size, num_steps):
    """Iterate on the raw data, a list of poems (with words converted to ids)
    This generates batch_size pointers into the raw text data, and allows
    minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls, should be a divisor of poem_length-1.
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    
    # Each Element of prepared_poems is a triple (themes,weights,poem),
    # where themes is a list of possible themes associated to the poem
    # weights is a list (of equal length), of how confident we are that the 
    # theme is associated with the poem.
        
    # Shuffle the data:
    prepared_poems=np.random.permutation(prepared_poems)
    
    #turn the poems into an array
    poem_array=np.array([p[2] for p in prepared_poems])
    

    data_len,poem_length = poem_array.shape
    # num_steps: int, the number of unrolls, should be a divisor of poem_length+1.
    assert poem_length % num_steps==1,"Poem_Length - 1,%d should be divisible by the num_steps, %d" %(poem_length-1,num_steps)
    batch_len = data_len // batch_size

    for j in range(batch_len):
        # Load the current batch
        current_batch=poem_array[j*batch_size:(j+1)*batch_size,:]
        # Randomly pick the corresponding themes
        themes=[np.random.choice(prepared_poems[i][0],p=prepared_poems[i][1]) for i in range(j*batch_size,(j+1)*batch_size)]
        for i in range(poem_length//num_steps):
            x=current_batch[:,i*num_steps:(i+1)*num_steps]
            y = current_batch[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x,y,themes,i==0)