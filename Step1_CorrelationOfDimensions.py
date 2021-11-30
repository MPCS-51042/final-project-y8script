import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import scipy
from math import ceil


d1 = pd.read_excel("Exp1MemorabilityScore.xlsx",index_col=0) # memorability data from experiment 1
d2 = pd.read_excel("Exp2MemorabilityScore.xlsx",index_col=0) # memorability data from experiment 2
d3 = pd.read_excel("Concretenss_Ratings.xlsx",index_col=1) # Concreteness ratings
d4 = pd.read_csv("cvaw4.csv",index_col=1) # Emotional valence ratings
d5 = pd.read_csv("similarity_matrix.csv",index_col=0) # Semantic similarity matrix
d6 = pd.read_excel("word_freq.xlsx",index_col=0) # Word frequency data
model_word2vec = KeyedVectors.load_word2vec_format("Exp1_Word2Vec_Embeddings.txt",binary=False)
model_FastText = KeyedVectors.load_word2vec_format('Exp1_FastText_Embedding.txt', binary=False)
model_Tencent = KeyedVectors.load_word2vec_format('Exp1_TencentEmbedding.txt', binary=False)

def get_emb_df_exp1(model):
    
    emb_exp1 = {}
    for i in range(len(d1.index)): 
        word = d1.index[i].strip("\'") 
        emb_exp1[word] = model.get_vector(word)
    emb_df_exp1 = pd.DataFrame(emb_exp1)
    emb_df_exp1 = emb_df_exp1.T
    #emb_df_exp1.to_csv('sgnswiki_exp1.csv', encoding='utf_8_sig')
    return emb_df_exp1
    
def get_emb_df_exp2(model):
    emb_exp2 = {}
    for i in range(len(d2.index)): 
        word = d2.index[i].strip("\'") 
        emb_exp2[word] = model.get_vector(word)

    emb_df_exp2 = pd.DataFrame(emb_exp2)
    emb_df_exp2 = emb_df_exp2.T
    #emb_df_exp2.to_csv('sgnswiki_exp2.csv', encoding='utf_8_sig')
    return emb_df_exp2

def score_correlation(emb_1,emb_2):
    corrvector_1 = []
    n_dim = len(emb_1.loc['坦白'])
    for index in range(n_dim):
        xx = emb_1[[index]].squeeze()
        x2 = [x for x in xx]
        yy = d1[['recog_score']].squeeze()
        y2 = [y for y in yy]
        corrvector_1.append(scipy.stats.spearmanr(x2, y2).correlation)
        
    corrvector_2 = []
    for index in range(n_dim):
        xx = emb_2[[index]].squeeze()
        x2 = [x for x in xx]
        yy = d2[['recog_score']].squeeze()
        y2 = [y for y in yy]
        corrvector_2.append(scipy.stats.spearmanr(x2, y2).correlation)
        
    print("Correlation between recog memorability and target-associative memorability:")
    print(scipy.stats.spearmanr(corrvector_1,corrvector_2).correlation)
    return corrvector_1, corrvector_2

def plot_dimensions(corrvector_1,corrvector_2):
    n_dim = len(corrvector_1)
    n_plots = ceil(n_dim/50)
    for i in range(1,n_plots+1):
        plt.subplot(ceil(n_plots/2),2, i)
        bar_width = 0.35
        ran = np.arange(50*(i-1),50*i)
        index = np.arange(50*(i-1),50*i)
        plt.bar(index, np.array(corrvector_1)[ran], bar_width, color='#0072BC')
        plt.bar(index + bar_width, np.array(corrvector_2)[ran], bar_width, color='#ED1C24')
    plt.show()

'''

'''

if __name__ == '__main__':
    for i in range(3):
        if i == 0:
            emb_df_exp1 = get_emb_df_exp1(model_word2vec)
            emb_df_exp2 = get_emb_df_exp2(model_word2vec)
            print("Word embeeding: Word2Vec")
        elif i == 1:
            emb_df_exp1 = get_emb_df_exp1(model_FastText)
            emb_df_exp2 = get_emb_df_exp2(model_FastText)
            print("Word embeeding: FastText")
        elif i == 2:
            emb_df_exp1 = get_emb_df_exp1(model_Tencent)
            emb_df_exp2 = get_emb_df_exp2(model_Tencent)
            print("Word embeeding: Tencent")                        
        corrvector_recog,corrvector_target = score_correlation(emb_df_exp1,emb_df_exp2)
        plot_dimensions(corrvector_recog,corrvector_target)
        print('Correlation between the dimension weights between exp1 and exp2:')
        print(scipy.stats.spearmanr(corrvector_recog,corrvector_target))
        print("\n")