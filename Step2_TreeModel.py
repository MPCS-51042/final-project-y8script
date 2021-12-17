# Use regression tree model to predict
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import re
from collections import Counter
from statistics import mean
from Step1_CorrelationOfDimensions import *

def Regression_tree(score,emb,split_ratio):
    dim_counts = Counter();
    correlations = []
    for times in range(200):
        yy = score[['recog_score']]
        xx = emb
        x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=split_ratio)
        clf = DecisionTreeRegressor(max_leaf_nodes=30)
        clf = clf.fit(x_train,y_train)
        
        y_pred = clf.predict(x_test)   
        
        text_representation = tree.export_text(clf)
        pattern = re.compile(r'(?<=feature_)\d*')
        important_dims = np.unique(pattern.findall(text_representation))
        for dim_num in important_dims:
            dim_counts[dim_num] += 1
        if times == 100:
            
            X_test = np.linspace(0, 1, y_pred.size)
            y_pred = list(y_pred)
            y_test = list(y_test.values)
            y_pred_sorted = [x for _,x in sorted(zip(y_test,y_pred))]
            y_test_sorted = sorted(y_test)
            plt.plot(X_test, y_pred_sorted, label="Model")
            plt.plot(X_test, y_test_sorted, label="True score")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim((0, 1))
            plt.legend(loc="best")
            plt.show()
        correlations.append(scipy.stats.spearmanr(y_pred,y_test).correlation)
    return clf, dim_counts, correlations
        
def get_important_dimensions(counter):
    dims_list = []
    for key in counter.keys():
        if counter[key] > 100:
            dims_list.append(key)
    return dims_list


if __name__ == '__main__':
    for i in range(3):
        if i == 0:
            emb_df_exp1 = get_emb_df_exp1(model_word2vec)
            emb_df_exp2 = get_emb_df_exp2(model_word2vec)
            print("Word embedding: Word2Vec")
            tree_exp1,dims_exp1,corrs_exp1 = Regression_tree(d1[['recog_score']],emb_df_exp1,0.1)              
            tree_exp2,dims_exp2,corrs_exp2 = Regression_tree(d2[['recog_score']],emb_df_exp2,0.1)
            print("Test-predict correlation for exp1:")
            print(mean(corrs_exp1))
            print("Test-predict correlation for exp2:")
            print(mean(corrs_exp2))
            print("Important dimensions for exp1:")
            print(get_important_dimensions(dims_exp1))
            print("Important dimensions for exp1:")
            print(get_important_dimensions(dims_exp2))
        elif i == 1:
            emb_df_exp1 = get_emb_df_exp1(model_FastText)
            emb_df_exp2 = get_emb_df_exp2(model_FastText)
            print("Word embedding: FastText")
            tree_exp1,dims_exp1,corrs_exp1 = Regression_tree(d1[['recog_score']],emb_df_exp1,0.1)              
            tree_exp2,dims_exp2,corrs_exp2 = Regression_tree(d2[['recog_score']],emb_df_exp2,0.1)
            print("Test-predict correlation for exp1:")
            print(mean(corrs_exp1))
            print("Test-predict correlation for exp2:")
            print(mean(corrs_exp2))
            print("Important dimensions for exp1:")
            print(get_important_dimensions(dims_exp1))
            print("Important dimensions for exp1:")
            print(get_important_dimensions(dims_exp2))
        elif i == 2:
            emb_df_exp1 = get_emb_df_exp1(model_Tencent)
            emb_df_exp2 = get_emb_df_exp2(model_Tencent)
            print("Word embedding: Tencent")
            tree_exp1,dims_exp1,corrs_exp1 = Regression_tree(d1[['recog_score']],emb_df_exp1,0.1)              
            tree_exp2,dims_exp2,corrs_exp2 = Regression_tree(d2[['recog_score']],emb_df_exp2,0.1)
            print("Test-predict correlation for exp1:")
            print(mean(corrs_exp1))
            print("Test-predict correlation for exp2:")
            print(mean(corrs_exp2))
            print("Important dimensions for exp1:")
            print(get_important_dimensions(dims_exp1))
            print("Important dimensions for exp1:")
            print(get_important_dimensions(dims_exp2))                