#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[10]:


# 데이터 준비

iris = datasets.load_iris()
X = iris.data[:,(2,3)]  #iris dataset에서 꽃잎 길이와 너비만 불러온다.
y = (iris['target']==2).astype(np.float64)  #Virginica


# ### 1. 선형 SVM 분류기

# In[11]:


svm_clf = Pipeline([
    ('scaler', StandardScaler()),  #스케일 변경
    ('linear_scv',LinearSVC(C=1,loss='hinge')), 
])

svm_clf.fit(X,y) #svm 모델 훈련


# In[12]:


svm_clf.predict([[5.5, 1.7]]) 


# 꽃잎 길이와 너비 값을 넣었을 때, Virginica 품종인지/아닌지 예측할 수 있다. 확률은 나오지 않는다.

# ### 2. SVC에서 선형 커널 사용

# In[16]:


from sklearn.svm import SVC


# In[20]:


svc_clf = SVC(kernel='linear', C=1)
svc_clf.fit(X,y)


# In[21]:


svc_clf.predict([[5.5,1.7]])


# ### 3. SGD 분류기

# In[19]:


from sklearn.linear_model import SGDClassifier


# In[33]:


m = len(X)  #m 은 샘플 개수
C = 1 #C는 위와 동일하게 1로 설정. 이때의 C는 하이퍼파라미터.


# In[35]:


sgd_clf = SGDClassifier(loss='hinge', alpha=1/(m*C))
sgd_clf.fit(X,y)


# In[36]:


sgd_clf.predict([[5.5,1.7]])


# 
