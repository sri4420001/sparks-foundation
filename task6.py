import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
data3=pd.read_csv('Iris.csv')
data3
figcls,axcls =  plt.subplots(1,2)
axcls[0].boxplot(data3['SepalLengthCm'])
axcls[1].boxplot(data3['SepalWidthCm'])
axcls[0].set_title('SepalLengthCm')
axcls[1].set_title('SepalWidthCm')
plt.show()
figcls,axcls =  plt.subplots(1,2)
axcls[0].boxplot(data3['PetalLengthCm'])
axcls[1].boxplot(data3['PetalWidthCm'])
axcls[0].set_title('PetalLengthCm')
axcls[1].set_title('PetalWidthCm')
plt.show()
q1cls=data3['SepalWidthCm'].quantile(0.25)
q3cls=data3['SepalWidthCm'].quantile(0.75)
iqrcls=q3cls-q1cls
mincls=q1cls-(1*iqrcls)
maxcls=q3cls+(1*iqrcls)
print(mincls,maxcls)
data3=data3[data3['SepalWidthCm']>=mincls]
data3=data3[data3['SepalWidthCm']<=maxcls]
#data3
figcls,axcls =  plt.subplots(1,2)
axcls[0].boxplot(data3['SepalLengthCm'])
axcls[1].boxplot(data3['SepalWidthCm'])
axcls[0].set_title('SepalLengthCm')
axcls[1].set_title('SepalWidthCm')
plt.show()
figcls,axcls =  plt.subplots(1,2)
axcls[0].boxplot(data3['PetalLengthCm'])
axcls[1].boxplot(data3['PetalWidthCm'])
axcls[0].set_title('PetalLengthCm')
axcls[1].set_title('PetalWidthCm')
plt.show()
clsxtrain,clsxtest,clsytrain,clsytest=train_test_split(data3.iloc[:,1:-1].values,
                                                      data3.iloc[:,-1:],test_size=0.3,random_state=1)
cls1=DecisionTreeClassifier().fit(clsxtrain,clsytrain)
cls1
predcls=cls1.predict(clsxtest)
acc1 = round(metrics.accuracy_score(clsytest, predcls),3)
print('decision tree pred : ',acc1)
sda=tree.export_text(cls1,feature_names=['SepalLengthCm'	,'SepalWidthCm'	,'PetalLengthCm'	,'PetalWidthCm'	])
print(sda)
figcls = plt.figure(figsize=(21,16))
tree.plot_tree(cls1,feature_names=['SepalLengthCm'	,'SepalWidthCm'	,'PetalLengthCm'	,'PetalWidthCm'	],filled=True)
plt.show()