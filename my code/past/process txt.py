import matplotlib.pyplot as plt
import itertools
import math
import pandas as pd
import os
import numpy as np
import string
import torch

# ner_df = pd.read_csv('C:/Users/admin\\Desktop\\自然語言處理\\NER\\ner_dataset.csv')
# fp = open(r'C:\\Users\\admin\\Desktop\\自然語言處理\\project\\crf_basline.txt',encoding="utf-8")
# line = fp.readline()
 

# f = open(r'C:\\Users\\admin\\Desktop\\自然語言處理\\project\\crf_basline.txt',encoding="utf-8")
# lines = f.readlines()#读取全部内容
 
 
# for i in range(0,lines.__len__(),1): #(开始/左边界, 结束/右边界, 步长)
 
#     list = []  ## 空列表, 将第i行数据存入list中
#     for word in lines[i].split():
#          word=word.strip(string.whitespace)
#          list.append(word);
#     print(list)

import re
import pandas as pd
import nltk

def parse_document(document):
   document = re.sub('\n', ' ', document)
   if isinstance(document, str):
       document = document
   else:
       raise ValueError('Document is not string!')
   document = document.strip()
   sentences = nltk.sent_tokenize(document)
   sentences = [sentence.strip() for sentence in sentences]
   return sentences

# sample document
text = """
FIFA was founded in 1904 to oversee international competition among the national associations of Belgium, 
Denmark, France, Germany, the Netherlands, Spain, Sweden, and Switzerland. Headquartered in Zürich, its 
membership now comprises 211 national associations. Member countries must each also be members of one of 
the six regional confederations into which the world is divided: Africa, Asia, Europe, North & Central America 
and the Caribbean, Oceania, and South America.
"""

# tokenize sentences
sentences = parse_document(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# tag sentences and use nltk's Named Entity Chunker
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
ne_chunked_sents = [nltk.ne_chunk(tagged) for tagged in tagged_sentences]
# extract all named entities
named_entities = []
for ne_tagged_sentence in ne_chunked_sents:
   for tagged_tree in ne_tagged_sentence:
       # extract only chunks having NE labels
       if hasattr(tagged_tree, 'label'):
           entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #get NE name
           entity_type = tagged_tree.label() # get NE category
           named_entities.append((entity_name, entity_type))
           # get unique named entities
           named_entities = list(set(named_entities))

# store named entities in a data frame
entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type'])
# display results
print(entity_frame)




