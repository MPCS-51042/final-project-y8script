# Linear Regression model for predicting memorability
from zhconv import convert
from math import log
from Step1_CorrelationOfDimensions import *
import statsmodels.api as sm

concret = []
logfreq = []
evalence = []
mean_sim = []
const = []

for item in d1.index:
    word = item.strip("\'")
    if word in d3.index:
        concret.append(d3.loc[word][1])
    else:
        concret.append(d3.mean(0)[1])
    if convert(word,'zh-hant') in d4.index:
        evalence.append(d4.loc[convert(word,'zh-hant')][1])
    else:
        evalence.append(d4.mean(0)[1])
    logfreq.append(log(d6.loc[word][1]))
    mean_sim.append(d5.mean(0)[word])
    const.append(1)

x = []
for i in range(len(concret)):
    x.append([concret[i], evalence[i], logfreq[i], mean_sim[i], const[i]])
exp1_memorability = d1['recog_score'].tolist()

print("Regression result for exp1:")
model = sm.OLS(exp1_memorability, x)
results = model.fit()
print(results.summary())

concret = []
logfreq = []
evalence = []
mean_sim = []
const = []

for item in d2.index:
    word = item.strip("\'")
    if word in d3.index:
        concret.append(d3.loc[word][1])
    else:
        concret.append(d3.mean(0)[1])
    if convert(word,'zh-hant') in d4.index:
        evalence.append(d4.loc[convert(word,'zh-hant')][1])
    else:
        evalence.append(d4.mean(0)[1])
    logfreq.append(log(d6.loc[word][1]))
    mean_sim.append(d5.mean(0)[word])
    const.append(1)

x = []
for i in range(len(concret)):
    x.append([concret[i], evalence[i], logfreq[i], mean_sim[i], const[i]])
exp2_memorability = d2['recog_score'].tolist()

print("\nRegression result for exp2:")
model = sm.OLS(exp2_memorability, x)
results = model.fit()
print(results.summary())

