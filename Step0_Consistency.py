import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import scipy
from math import ceil
from sklearn.model_selection import train_test_split

d1 = pd.read_excel("Exp1MemorabilityScore.xlsx",index_col=0)
d2 = pd.read_excel("Exp2MemorabilityScore.xlsx",index_col=0)
model_word2vec = KeyedVectors.load_word2vec_format("Exp1_Word2Vec_Embeddings.txt",binary=False)
model_FastText = KeyedVectors.load_word2vec_format('Exp1_FastText_Embedding.txt', binary=False)
model_Tencent = KeyedVectors.load_word2vec_format('Exp1_TencentEmbedding.txt', binary=False)

trials = pd.read_excel('Exp2Trials.xlsx')

# Compute semantic similarity matrix of words
similarity = {}
for x in d1.index:
    x_word = x.strip("\'")
    similarity[x_word] = {}

for x in d1.index:
    x_word = x.strip("\'")
    for y in d1.index:
        y_word = y.strip("\'")
        similarity[x_word][y_word] = model_Tencent.similarity(x_word,y_word)

sdf = pd.DataFrame(similarity)
sdf.to_csv("similarity_matrix.csv")

# Consistency check - split-half correlation
correlations = []
sub_ids = np.unique(trials[['subID']]).tolist()

for times in range(100):
    target_total = {}
    target_right = {}
    target_wrong = {}
    target_correct = {}
    for x in d2.index:
        x_word = x.strip("\'")
        target_total[x_word] = {1:0,2:0}
        target_right[x_word] = {1:0,2:0}
        target_wrong[x_word] = {1:0,2:0}
        target_correct[x_word] = {1:0,2:0}
    
    group1,group2 = train_test_split(sub_ids,test_size = 0.5)
    
    for row in trials.values:
        row_word = row[7].strip("\'")
        if row[0] in group1:
            target_total[row_word][1] += 1
            target_right[row_word][1] += row[4]
            target_wrong[row_word][1] += (1 - row[4])
        else:
            target_total[row_word][2] += 1
            target_right[row_word][2] += row[4]
            target_wrong[row_word][2] += (1 - row[4])        
    
    for key in target_right:
        target_correct[key][1] = target_right[key][1]/target_total[key][1]
        target_correct[key][2] = target_right[key][2]/target_total[key][2]
    
    score_1 = []
    score_2 = []
    for key in target_correct:
        score_1.append(target_correct[key][1])
        score_2.append(target_correct[key][2])
    
    correlations.append(scipy.stats.spearmanr(score_1,score_2).correlation)

plt.hist(correlations)
plt.xlabel("Correlation between group1 and group2")
plt.ylabel("Times")
plt.show()