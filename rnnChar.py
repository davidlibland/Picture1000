#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

batch_size=4
#num_steps=8
seq_length=128
learning_rate=0.01

#vocab_size=64
hidden_size=128
max_steps=5000

print("-"*30)
print("Data Input")
print("-"*30)

# data I/O
data = open('haiku.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


print("-"*30)
print("initialization")
print("-"*30)

sess=tf.InteractiveSession()


# Create a list of input/target placeholders
inputText = [tf.placeholder(tf.int32, 1) for _ in range(seq_length)] #batch_size x num_steps
targetText = [tf.placeholder(tf.int32, 1) for _ in range(seq_length)] #batch_size x num_steps


# convert to 1-hot-labeling input/target placeholders
embedding=tf.diag([1.]*vocab_size, name="embedding")
inputs,targets=[],[]
for i in range(seq_length):
    inputs.append(tf.gather(embedding,inputText[i],name="input"+str(i)))
    targets.append(tf.gather(embedding,targetText[i],name="input"+str(i)))
    

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
Wxh = weight_variable([vocab_size,hidden_size])
Whh = weight_variable([hidden_size,hidden_size])
bh = bias_variable([hidden_size])

Who = weight_variable([hidden_size,vocab_size])
bo = bias_variable([vocab_size])
    
#Create the list of hidden variables, output variables, and loss
hidden=[tf.nn.sigmoid(tf.matmul(inputs[0],Wxh)+bh)]
output=[tf.nn.softmax(tf.matmul(hidden[0],Who)+bo)]
softmax_cross_entropy=[tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(hidden[0],Who)+bo,targets[0]))]
for i in range(1,seq_length):
    hidden.append(tf.nn.sigmoid(tf.matmul(inputs[i],Wxh)+tf.matmul(hidden[i-1],Whh)+bh))
    output.append(tf.nn.softmax(tf.matmul(hidden[i],Who)+bo))
    softmax_cross_entropy.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(hidden[i],Who)+bo,targets[i])))
    

with tf.name_scope('total'):
    total_cross_entropy=softmax_cross_entropy[0]/seq_length
    for i in range(1,seq_length):
        total_cross_entropy+=softmax_cross_entropy[i]/seq_length
tf.scalar_summary('cross entropy', total_cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        total_cross_entropy)
        
        
 # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./tmp/HaikuTrainSum',
                                        sess.graph)
test_writer = tf.train.SummaryWriter('./tmp/testSum')



#Sampling code
# Create an input placeholder for sampling purposes
inputTextSamp=tf.placeholder(tf.int32, 1,name="SampleTextInput")
inputSamp=tf.gather(embedding,inputTextSamp)
# Create a hidden variable for sampling purposes
hiddenSamp=tf.Variable(tf.constant(0.,shape=[1,hidden_size]),name="SampleHiddenVariable")
# connect them
newHiddenSamp=tf.nn.sigmoid(tf.matmul(inputSamp,Wxh)+tf.matmul(hiddenSamp,Whh)+bh)
outputSamp=tf.nn.softmax(tf.matmul(newHiddenSamp,Who)+bo)
# create a reassignment op:
updateHidden=tf.assign(hiddenSamp,newHiddenSamp)
def sample(length):
    ix=np.random.choice(range(vocab_size))
    ixes = [ix]
    for i in range(length):
        out = outputSamp.eval(feed_dict={inputTextSamp:[ix]})
        updateHidden.eval(feed_dict={inputTextSamp:[ix]})
        ix = np.random.choice(range(vocab_size), p=out.ravel())
        ixes.append(ix)
    return ixes


tf.initialize_all_variables().run()
        

print("-"*30)
print("training")
print("-"*30)
p=0
for i in range(max_steps):
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    feed_dict={}
    #for j=range(batch_size):
    if p+seq_length+1 >= len(data): 
        p = 0 # go from start of data
    inputsFeed = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targetsFeed = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    for j in range(seq_length):
        feed_dict[inputText[j]]=[inputsFeed[j]]
        feed_dict[targetText[j]]=[targetsFeed[j]]
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
    train_writer.add_summary(summary, i)
    if i%100==0: # output a sample of text
        print('Iteration: %i' %i)
        sample_ix = sample(200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))
    p += seq_length # move data pointer
        