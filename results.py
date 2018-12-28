# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:02:01 2018

@author: lawle
"""

# In[1]:

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

# In[2]:

# Load the data
train = pd.read_csv("labeledTrainData.tsv", delimiter="\t")

test_text = []
test_label = []

test_flist = os.listdir('./test/pos')
for i in test_flist:
    buffer = open('./test/pos/' + str(i), encoding = 'utf-8')
    test_text.append(buffer.read())
    test_label.append(1)
    
test_flist = os.listdir('./test/neg')
for i in test_flist:
    buffer = open('./test/neg/' + str(i), encoding = "utf-8")
    test_text.append(buffer.read())
    test_label.append(0)


train_label = []
for i in train.sentiment:
    if i == 0:
        train_label.append(1)
    elif i == 1:
        train_label.append(0)
train_label = np.asarray(train_label, dtype=np.int32)

nltk.download('stopwords')

# # Clean and Format the Data

# In[8]:

def clean_text(text, remove_stopwords=True):
    '''Clean the text, with the option to remove stopwords'''
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"   ", " ", text) # Remove any extra spaces
    text = re.sub(r"  ", " ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Return a list of words
    return(text)


# Clean the training and testing reviews

# In[9]:

train_clean = []
for review in train.review:
    train_clean.append(clean_text(review))

# In[10]:

test_clean = []
for review in test_text:
    test_clean.append(clean_text(review))


# In[13]:

# Tokenize the reviews
all_reviews = train_clean + test_clean
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_reviews)
print("Fitting is complete.")

train_seq = tokenizer.texts_to_sequences(train_clean)
print("train_seq is complete.")

test_seq = tokenizer.texts_to_sequences(test_clean)
print("test_seq is complete")


# In[14]:

# Find the number of unique tokens
word_index = tokenizer.word_index
print("Words in index: %d" % len(word_index))

# In[19]:

# Pad and truncate the questions so that they all have the same length.
max_review_length = 200
train_pad = pad_sequences(train_seq, maxlen = max_review_length)
print("train_pad is complete.")

test_pad = pad_sequences(test_seq, maxlen = max_review_length)
print("test_pad is complete.")

# In[21]:

# Creating the training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(train_pad, train.sentiment, test_size = 0.1, random_state = 2)
x_train_adv, x_valid_adv, y_train_adv, y_valid_adv = train_test_split(train_pad, train_label, test_size = 0.1, random_state = 2)
x_test = test_pad
y_test = np.asarray(test_label)

# In[23]:

def get_batches(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

# In[88]:
def get_a_cell(lstm_size, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, reuse=tf.AUTO_REUSE)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop
   

# In[89]:
    
def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers, 
              dropout, learning_rate, multiple_fc, fc_units):
    
    tf.reset_default_graph()
    
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    
    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    with tf.variable_scope('embeddings') as emb:
        embeddings = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embeddings, inputs)
                
    with tf.variable_scope('lstm') as lstm:
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(lstm_size, keep_prob) for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                 initial_state=initial_state)     
            
    with tf.variable_scope('fully_connected') as fcn:
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        
        dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                                                  num_outputs = fc_units,
                                                  activation_fn = tf.sigmoid,
                                                  weights_initializer = weights,
                                                  biases_initializer = biases)
        dense = tf.contrib.layers.dropout(dense, keep_prob)
                
    with tf.variable_scope('predictions') as lastfcn:
        predictions = tf.contrib.layers.fully_connected(dense, 
                                                        num_outputs = 1, 
                                                        activation_fn=tf.sigmoid,
                                                        weights_initializer = weights,
                                                        biases_initializer = biases)
        tf.summary.histogram('predictions', predictions)
        
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('cost', cost)
    
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    merged = tf.summary.merge_all()
    
    export_nodes = ['inputs', 'labels', 'keep_prob', 'embeddings', 'initial_state', 'final_state','accuracy',
                    'predictions', 'cost', 'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph

# In[92]:

def train(model, epochs, log_string):
    
    saver = tf.train.Saver()
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Used to determine when to stop the training early
    valid_loss_summary = []
    
    # Keep track of which batch iteration is being trained
    iteration = 0
    stop_early = 0
    
    print()
    print("Training Model: {}".format(log_string))
    
    train_writer = tf.summary.FileWriter('./logs/3/train/{}'.format(log_string), sess.graph)
    valid_writer = tf.summary.FileWriter('./logs/3/valid/{}'.format(log_string))
    
    for e in range(epochs):
        state = sess.run(model.initial_state)
        
        # Record progress with each epoch
        train_loss = []
        train_acc = []
        val_acc = []
        val_loss = []
    
        with tqdm(total=len(x_train)) as pbar:
            for _, (x, y) in enumerate(get_batches(x_train, y_train, batch_size), 1):
                feed = {model.inputs: x,
                        model.labels: y[:, None],
                        model.keep_prob: dropout,
                        model.initial_state: state}
                summary, loss, acc, state, _ = sess.run([model.merged, 
                                                         model.cost, 
                                                         model.accuracy, 
                                                         model.final_state, 
                                                         model.optimizer], 
                                                        feed_dict=feed)                
                
                # Record the loss and accuracy of each training batch
                train_loss.append(loss)
                train_acc.append(acc)
                
                # Record the progress of training
                train_writer.add_summary(summary, iteration)
                
                iteration += 1
                pbar.update(batch_size)
        
        # Average the training loss and accuracy of each epoch
        avg_train_loss = np.mean(train_loss)
        avg_train_acc = np.mean(train_acc) 
    
        val_state = sess.run(model.initial_state)
        with tqdm(total=len(x_valid)) as pbar:
            for x, y in get_batches(x_valid, y_valid, batch_size):
                feed = {model.inputs: x,
                        model.labels: y[:, None],
                        model.keep_prob: 1,
                        model.initial_state: val_state}
                summary, batch_loss, batch_acc, val_state = sess.run([model.merged, 
                                                                      model.cost, 
                                                                      model.accuracy, 
                                                                      model.final_state], 
                                                                     feed_dict=feed)
                
                val_loss.append(batch_loss)
                val_acc.append(batch_acc)
                pbar.update(batch_size)
        
        avg_valid_loss = np.mean(val_loss)    
        avg_valid_acc = np.mean(val_acc)
        valid_loss_summary.append(avg_valid_loss)
        
        valid_writer.add_summary(summary, iteration)
    
        print("Epoch: {}/{}".format(e, epochs),
              "Train Loss: {:.3f}".format(avg_train_loss),
              "Train Acc: {:.3f}".format(avg_train_acc),
              "Valid Loss: {:.3f}".format(avg_valid_loss),
              "Valid Acc: {:.3f}".format(avg_valid_acc))
    
        if avg_valid_loss > min(valid_loss_summary):
            print("No Improvement.")
            stop_early += 1
            if stop_early == 3:
                break   
        
        else:
            print("New Record!")
            stop_early = 0
            checkpoint = './model/sentiment_{}.ckpt'.format(log_string)
            saver.save(sess, checkpoint)
    
    return np.array(sess.run(model.embeddings))
              
# In[93]:

# The default parameters of the model
n_words = len(word_index)
embed_size = 300
batch_size = 250
lstm_size = 128
num_layers = 2
dropout = 0.75
learning_rate = 0.001
epochs = 50
multiple_fc = False
fc_units = 256 

# In[92]:
log_string = 'ru={},fcl={},fcu={}'.format(lstm_size, multiple_fc, fc_units)

model = build_rnn(n_words = n_words, 
                  embed_size = embed_size,
                  batch_size = batch_size,
                  lstm_size = lstm_size,
                  num_layers = num_layers,
                  dropout = dropout,
                  learning_rate = learning_rate,
                  multiple_fc = multiple_fc,
                  fc_units = fc_units)      

embed_table = train(model, epochs, log_string)
ckpt_loc = './model/sentiment_ru=128,fcl=False,fcu=256.ckpt'

# In[92]:
def make_predictions(checkpoint):
    test_acc = []

    model = build_rnn(n_words = n_words, 
                      embed_size = embed_size,
                      batch_size = batch_size,
                      lstm_size = lstm_size,
                      num_layers = num_layers,
                      dropout = dropout,
                      learning_rate = learning_rate,
                      multiple_fc = multiple_fc,
                      fc_units = fc_units) 
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # Load the model
        saver.restore(sess, checkpoint)
        test_state = sess.run(model.initial_state)
        with tqdm(total=len(x_test)) as pbar:
            for _, (x, y) in enumerate(get_batches(x_test, y_test, batch_size), 1):
                feed = {model.inputs: x,
                        model.labels: y[:, None],
                        model.keep_prob: 1,
                        model.initial_state: test_state}
                batch_acc = sess.run(model.accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
                pbar.update(batch_size)
                
    avg_test_acc = np.mean(test_acc)
    
    return avg_test_acc

# =============================================================================
# acc = make_predictions('./model/sentiment_ru=128,fcl=False,fcu=256.ckpt')
# acc 
# =============================================================================
    
# In[140]:

tf.reset_default_graph()

with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    
with tf.name_scope('labels'):
    labels = tf.placeholder(tf.int32, [None, None], name='labels')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE) as emb:
    embeddings = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embeddings, inputs)
            
with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE) as lstm:
    cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)     
        
with tf.variable_scope('fully_connected', reuse=tf.AUTO_REUSE) as fcn:
    weights = tf.truncated_normal_initializer(stddev=0.1)
    biases = tf.zeros_initializer()
    
    dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                                              num_outputs = fc_units,
                                              activation_fn = tf.sigmoid,
                                              weights_initializer = weights,
                                              biases_initializer = biases)
    dense = tf.contrib.layers.dropout(dense, keep_prob)
            
with tf.variable_scope('predictions', reuse=tf.AUTO_REUSE) as lastfcn:
    predictions = tf.contrib.layers.fully_connected(dense, 
                                                    num_outputs = 1, 
                                                    activation_fn=tf.sigmoid,
                                                    weights_initializer = weights,
                                                    biases_initializer = biases)
    tf.summary.histogram('predictions', predictions)
    
with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(labels, predictions)
    tf.summary.scalar('cost', cost)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

sess=tf.Session()
saver = tf.train.Saver()
saver.restore(sess, ckpt_loc)
sess.run('fully_connected/fully_connected/weights:0')

adv = tf.Variable(tf.random_uniform((20, embed_size), -0.3, 0.3), name='advtexts')
advtexts = [adv for i in range(batch_size)]
embed = tf.concat([embed, advtexts], 1)
outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)     

with tf.variable_scope('fully_connected', reuse=tf.AUTO_REUSE):
    dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                                              num_outputs = fc_units,
                                              activation_fn = tf.sigmoid,
                                              weights_initializer = weights,
                                              biases_initializer = biases)
    dense = tf.contrib.layers.dropout(dense, keep_prob)
  
with tf.variable_scope('predictions', reuse=tf.AUTO_REUSE):        
    predictions = tf.contrib.layers.fully_connected(dense, 
                                                    num_outputs = 1, 
                                                    activation_fn=tf.sigmoid,
                                                    weights_initializer = weights,
                                                    biases_initializer = biases)
    
cost = tf.losses.mean_squared_error(labels, predictions)
correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost, var_list = [advtexts])
    
var_list = [var for var in tf.global_variables() if 'train' in var.name]
var_list2 = [var for var in tf.global_variables() if 'advtexts' in var.name]
sess.run(tf.variables_initializer(var_list))
sess.run(tf.variables_initializer(var_list2))

valid_loss_summary = []
stop_early=0
for e in range(epochs):
    train_loss = []
    train_acc = []
    val_acc = []
    val_loss = []
    
    state = sess.run(initial_state)
    
    with tqdm(total=len(x_train)) as pbar:
        for _, (x, y) in enumerate(get_batches(x_train_adv, y_train_adv, batch_size), 1):
            feed = {inputs: x,
                    labels: y[:, None],
                    keep_prob: dropout,
                    initial_state: state}
            loss, acc, state,  _ = sess.run([cost,accuracy,final_state, optimizer],feed_dict=feed)                
            
            # Record the loss and accuracy of each training batch
            train_loss.append(loss)
            train_acc.append(acc)
            
            pbar.update(batch_size)
    
    # Average the training loss and accuracy of each epoch
    avg_train_loss = np.mean(train_loss)
    avg_train_acc = np.mean(train_acc) 

    val_state = sess.run(initial_state)
    with tqdm(total=len(x_valid)) as pbar:
        for x, y in get_batches(x_valid_adv, y_valid_adv, batch_size):
            feed = {inputs: x,
                    labels: y[:, None],
                    keep_prob: 1,
                    initial_state: val_state}
            batch_loss, batch_acc, val_state = sess.run([cost,accuracy,final_state], feed_dict=feed)
            
            # Record the validation loss and accuracy of each epoch
            val_loss.append(batch_loss)
            val_acc.append(batch_acc)
            pbar.update(batch_size)
    
    # Average the validation loss and accuracy of each epoch
    avg_valid_loss = np.mean(val_loss)    
    avg_valid_acc = np.mean(val_acc)
    valid_loss_summary.append(avg_valid_loss)
    
    # Print the progress of each epoch
    print("Epoch: {}/{}".format(e, epochs),
          "Train Loss: {:.3f}".format(avg_train_loss),
          "Train Acc: {:.3f}".format(avg_train_acc),
          "Valid Loss: {:.3f}".format(avg_valid_loss),
          "Valid Acc: {:.3f}".format(avg_valid_acc))

    # Stop training if the validation loss does not decrease after 3 epochs
    if avg_valid_loss > min(valid_loss_summary):
        print("No Improvement.")
        stop_early += 1
        if stop_early == 5:
            break   
    
    # Reset stop_early if the validation loss finds a new low
    # Save a checkpoint of the model
    else:
        print("New Record!")
        stop_early = 0

advtext_w = np.array(sess.run(adv))
emb_distances = tf.matmul(
        tf.nn.l2_normalize(embed_table, dim=1),
        tf.nn.l2_normalize(advtext_w, dim=1),
        transpose_b=True)

token_ids = tf.argmax(emb_distances, axis=0)
idss = sess.run(token_ids)

adv_text = []
for i in idss:
    for word, idx in word_index.items():
        if idx == i:
            adv_text.append(word)

print('adv text: ', adv_text)
print('origin test acc: ', make_predictions(ckpt_loc))
            
new_test = []
for i in range(len(test_seq)):
    new_test.append(np.append(idss,test_seq[i]))
    
test_pad = pad_sequences(new_test, 200)
x_test = test_pad
y_test = np.asarray(test_label)

print('adv test acc: ', make_predictions(ckpt_loc))
print('adv_weight: ', advtext_w)
    
