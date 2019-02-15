import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

p3 = pd.read_csv("Dataset_model.csv", sep=";",header=0, decimal=",")
qtd_obs = 2500
qtd_with_second = 30

var = []
for j in range(0, 10):
    var.append('V' + str(j))

p3.columns = var

tt2 = np.zeros((qtd_obs, 9))
levels = np.unique(p3['V9'])

cont = -1

for j in levels:
    cont = cont + 1
    cont1 = -1
    m = p3['V9'] == j
    p3_sub = p3[m]
    nv3 = np.unique(p3_sub['V8'])
    p3_sub = p3_sub.drop(columns=['V9'])
    media = p3_sub.groupby('V8').mean()
    desvio = p3_sub.groupby('V8').std()
    p3_sub = p3_sub.drop(columns=['V8'])
    cont3 = -1
    for i in range(0, qtd_obs):
        cont1 = cont1 + 1
        cont3 = cont3 + 1
        tt1 = np.zeros((qtd_with_second, 8))
        for k1 in range(0, 9):
            if k1 < 8:
                if cont3 < (len(nv3) - 1):
                    tt1[:, k1] = np.int_(
                        np.random.normal(media.iloc[cont3, k1], desvio.iloc[cont3, k1], qtd_with_second))
                else:
                    tt1[:, k1] = np.int_(
                        np.random.normal(media.iloc[cont3, k1], desvio.iloc[cont3, k1], qtd_with_second))
                    cont3 = -1
            if k1 == 8:
                tt1 = pd.DataFrame(tt1)
                f = tt1.std()
                f1 = tt1.mean()
                f2 = tt1.median()
                f3 = tt1.quantile(0.25)
                f4 = tt1.quantile(0.75)
                # f5 = tt1.sum()
                # f6 = np.log(abs(tt1.sum())+0.01)
                d = np.hstack((f, f1, f2, f3, f4))
                d1 = np.zeros((1, 5 * 8))
                for ij in range(0, len(d)):
                    d1[0, ij] = d[ij]
        if cont1 == 0:
            p4 = d1
        else:
            p4 = np.vstack((p4, d1))
        # print(i)
    if cont == 0:
        p5 = pd.DataFrame(p4)
        p5['target'] = j
    else:
        p4 = pd.DataFrame(p4)
        p4['target'] = j
        p5 = np.vstack((p5, p4))
    print(str(j))

levels = np.unique(p3['V8'])

cont1 = -1

for j in levels:
    cont1 = cont1 + 1
    m = p3['V8'] == j
    p3_sub = p3[m]
    target = np.unique(p3_sub['V9'])
    p3_sub = p3_sub.drop(columns=['V8', 'V9'])
    f = p3_sub.std()
    f1 = p3_sub.mean()
    f2 = p3_sub.median()
    f3 = p3_sub.quantile(0.25)
    f4 = p3_sub.quantile(0.75)
    # f5 = p3_sub.sum()
    # f6 = np.log(abs(tt1.sum())+0.01)
    d = np.hstack((f, f1, f2, f3, f4))
    d1 = np.zeros((1, 5 * 8))
    for ij in range(0, len(d)):
        d1[0, ij] = d[ij]
    if cont1 == 0:
        p4 = d1
        p4 = pd.DataFrame(p4)
        p4['target'] = target
    else:
        d1 = pd.DataFrame(d1)
        d1['target'] = target
        p4 = np.vstack((p4, d1))

p5 = pd.DataFrame(p5)

y = p5.iloc[:, 40]
y = y.replace(['Open Hands', 'Fist', 'Cool', 'Ok', 'PeaceandLove', 'Indicator'], [0, 1, 2, 3, 4, 5])

var = []
for j in range(0, len(p5.columns)):
    var.append('V' + str(j))

p5.columns = var
p5 = p5.drop(columns=var[len(p5.columns) - 1])

X = p5.values
Y = y.values

seed = 7
num_trees = 700
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed, learning_rate=0.1)
# results = model_selection.cross_val_score(model, X, Y, cv=kfold)
model.fit(X, Y)
p4 = pd.DataFrame(p4)
pickle.dump(model, open('modelo_gbm_myo', 'wb'))


var = []
for j in range(0, len(p4.columns)):
    var.append('V' + str(j))

p4.columns = var
yt = p4.iloc[:, 40]
yt = yt.replace(['Open Hands', 'Fist', 'Cool', 'Ok', 'PeaceandLove', 'Indicator'], [0, 1, 2, 3, 4, 5])

p4 = p4.drop(columns=var[len(p4.columns) - 1])

pred = model.predict_proba(p4.values)
maxrow = pred.max(axis=1)
feature_importance = model.feature_importances_

gt = []
for i in range(0, len(pred)):
    gg1 = np.where(pred[i, :] == maxrow[i])
    gt.append(int(gg1[0]))

print(str(sum(np.where(yt == gt, 1, 0)) / len(p4)))
print(str(pd.crosstab(yt, np.int_(gt))))