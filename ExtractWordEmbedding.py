# Extract the vector representation for the set of words in exp1 and exp2
import pandas as pd
from gensim.models import KeyedVectors


d1 = pd.read_excel("Exp1MemorabilityScore.xlsx",index_col=0)
model_full = KeyedVectors.load_word2vec_format("sgns.wiki.words",binary=False)

words = []
for i in range(len(d1.index)): 
    word = d1.index[i].strip("\'")
    words.append(word)

kv = KeyedVectors(300)
vectors = [model_word2vec[w] for w in words]
kv.add_vectors(words, vectors)
kv.save_word2vec_format('Exp1_Word2Vec_Embeddings.txt')