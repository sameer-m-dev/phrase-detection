#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Importing necessary packages
import nltk
from nltk.corpus import stopwords 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import csv
import pandas as pd
import numpy as np


# In[30]:


# Reading Training file.
data = pd.read_csv("training_data.tsv", delimiter = '\t', encoding = 'utf-8')


# In[31]:


# Checking out any random sentence from the training data-set
sentence = data['sent'][51]
sentence


# In[32]:


stop_words = set(stopwords.words('english')) 


# In[33]:


# Tokenizing sentence into individual words and removing the stopwords
tokens = nltk.word_tokenize(sentence)
tokens = [w for w in tokens if not w in stop_words] 
len(tokens)


# In[34]:


# Parts of Speech(POS) Tagging : Every word is assigned a tag
tagged = nltk.pos_tag(tokens)
tagged


# In[35]:


# Nouns and Verb type words can be considered as "important words"
Imp_words = [w[0] for w in tagged if w[1].startswith('N') or w[1].startswith('V')]
Imp_words


# In[36]:


from nltk.util import ngrams

n=5
for i in range(1,n+1):
    output = list(ngrams(tokens, i))
    print (output,"\n")


# In[37]:


# Defining a function which takes a sentence as an input and returns important phrase

def extract(sentence):
    
    words = nltk.word_tokenize(sentence)
    words = [w for w in words if not w in stop_words] 
    nltk.pos_tag(words)
    
    # defining a chunk grammar, consisting of rules that indicate how sentences should be chunked.
    # NP chunk should be formed whenever the chunker finds optional verb type(VB), followed by optional RB, 
    # folllowed by Personal pronoun types(PRP), followed by optional Preposition(IN) , followed by  an optional determiner (DT)
    # followed by any number of adjectives (JJ) and then a noun (NN). 
    
    grammar = "NP: {<VB.*>?<RB>?<PRP.*>?<IN>?<DT>?<JJ.*>*<NN.*>+}"
    
    # Using this grammar, we create a chunk parser
    parser = nltk.RegexpParser(grammar)
    
    # Test it on our example sentence
    t = parser.parse(nltk.pos_tag(words))
    
    # Result is a tree 
    a = [s for s in t.subtrees() if s.label() == "NP"]
    
    c = []
    num = []
    
    # These keywords were not included as label in training dat, so don't consider here also
    key  = ["monday","tuesday", "wednesday", "thursday","friday","saturday","sunday","today","tomorrow","yesterday", "reminder", "remind", "th", "pm","am"]
    
    for i in range(len(a)):
        count=0
        phrase = ""
        for j in range(len(a[i])):
            if a[i][j][0].lower() in key:
                phrase = phrase
            else :
                phrase = phrase + str(a[i][j][0]) + " "
                count = count+1
        c.append(phrase)
        num.append(count)
    
    if(c==[] or max(num)<=1):
        return "Not Found"
    else :
        maxi = max(num)
        for i in range(len(num)):
            if(num[i]==maxi):
                return c[i].rstrip()


# In[38]:


#testig the function
print(sentence,"\n") 
print("Phrase  :   ", extract(sentence))


# In[39]:


# Reading file line by line 
with open("eval_data.txt", 'r+') as f:
    lines = [line.rstrip('\n') for line in f]
    
print (lines[154])


# In[40]:


#creating new csv file
with open('eval_data.csv', mode='w', newline='') as csv_file:
    fieldnames = ['sent', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    for i in range(len(lines)):
        writer.writerow({'sent':lines[i],'label':extract(lines[i])})


# In[41]:


#checking accuracy of the model on the training set
with open('eval_data.csv', mode='w', newline='', encoding = 'utf-8') as csv_file:
    fieldnames = ['sent', 'Given_label', 'Predicted_label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    count = 0
    for i in range(len(data)):
        writer.writerow({'sent':data['sent'][i], 'Given_label':data['label'][i], 'Predicted_label':extract(str(data['sent'][i]))})
        
        if str(data['label'][i]) == extract(str(data['sent'][i])):
            count = count+1
            
print ("Accuracy : ", (count/len(data))*100, "%")

