#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
lemmatizer = WordNetLemmatizer()

#1. Import and load the data file
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
HRI2_intents_file = open('F:\Symbi Project\A\Human Resources Intelligent Interaction (HRI2)\HRI2_intents.json').read()
HRI2_intents = json.loads(HRI2_intents_file)
#2. Data Preprocessing
for HRI2_intent in HRI2_intents['HRI2_intents']:
    for pattern in HRI2_intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, HRI2_intent['tag']))
        # add to our classes list
        if HRI2_intent['tag'] not in classes:
            classes.append(HRI2_intent['tag'])

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and HRI2_intents
print (len(documents), "documents")
# classes = HRI2_intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

#3. creating a pickle file to store the Python objects which we will use while predicting
pickle.dump(words,open('HRI2_words.pkl','wb'))
pickle.dump(classes,open('HRI2_classes.pkl','wb'))

#4. create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - HRI2_intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

#5. MODEL BUILDING

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons equal to number of HRI2_intents to predict output HRI2_intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model using Stochastic gradient descent with Nesterov accelerated gradient
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#6.fitting and saving the model 
HRI2_model = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('HRI2_model.h5', HRI2_model)

print("Model Created")


# In[ ]:




