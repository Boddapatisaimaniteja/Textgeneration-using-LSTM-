# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:16:30 2020

@author: bodda
"""
import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename='dinos.txt'
data=open(r'E:\Text generation\dinos.txt','r',encoding='utf-8').read()
data=data.lower()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)

chars=sorted(chars)

char_to_ix={ch:i for i,ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}

#PREPARING DATA
seq_length=100
datax=[]
datay=[]
for i in range(0,data_size-seq_length,1):
    seq_in=data[i:i+seq_length]
    seq_out=data[i+seq_length]
    datax.append([char_to_ix[ch] for ch in seq_in])
    datay.append([char_to_ix[seq_out]])

n_patterns=len(datax)

x=np.reshape(datax,(n_patterns,seq_length,1))
x=x/float(vocab_size)
y=np_utils.to_categorical(datay)   

'''
model=Sequential()
model.add(LSTM(256,input_shape=(x.shape[1],x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
  

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model.fit(x, y, epochs=50, batch_size=128, callbacks=callbacks_list)
'''
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(chars)), p = probas.ravel())
    return out 

from keras.models import load_model


filename = r'E:\Text generation\namegen.h5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

	

int_to_char = dict((i, c) for i, c in enumerate(chars))

start=5
pattern=[0]*100
for i in range(1000):
    
    x=np.reshape(pattern,(1,len(pattern),1))
    x=x/float(vocab_size)
    pred=model.predict(x,verbose=2)[0]
    #index=np.argmax(pred)
    index=sample(pred)
    result=int_to_char[index]
    seq_in=[int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern=pattern[1:len(pattern)]
print("\nDone.")


        