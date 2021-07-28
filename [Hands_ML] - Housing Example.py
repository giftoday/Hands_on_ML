#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('Hello World!')
#파이썬 시작할때면 꼭 해보고싶다.. 헬로 월드! 


# In[16]:


import os
import tarfile
import urllib


# In[34]:


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_PATH = os.path.join('datasets','housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url=HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok = True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
#데이터를 불러오는 함수 준비_1 - 데이터 다운받고 압축을 풀기


# In[35]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

#데이터를 불러오는 함수 준비_2 - csv로 불러오기


# In[36]:


fetch_housing_data()


# In[37]:


housing = load_housing_data()
housing.head()
# 데이터 불러오기


# In[38]:


housing.info()

total_bedrooms 특성 : 20433개만 non-null. 나머지는 Null 값이라는 의미.즉 207개의 구역은 이 특성을 갖고있지 않음.
ocean_proximity 만 뺴고 모두 float형태. 위 표를 보면 NEAR BAY가 반복되고 있음 -> 범주형이라 추측 가능. 

ocean_proximity에 어떤 값들이 있는지 확인해보자.
# In[39]:


housing.ocean_proximity.value_counts()
# value_counts 쓰면 각 카테고리와 숫자까지 반환시켜준다.


# In[40]:


housing.ocean_proximity.unique()
# Unique 를 쓰면 어떤 카테고리가 있는지만 나온다.


# In[41]:


housing.describe()
# describe 는 숫자형 특성의 요약 정보 보여줌. 
# total_bedrooms의 count 값이 20433으로 다른 값들보다 작은 것을 볼 수 있다.


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')

#그래프를 주피터 안에 그리도록 명령

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

# housing의 모든 숫자형 특성에 대한 히스토그램을 출력.

1. median_income의 x축을 보자. 2, 4, 6 등이 중간소득이 될 수 없음. 즉 데이터 전처리 과정이 있었음을 의미한다. 
-> 스케일 조정한 후, 0.5~15 사이로 줄인 것. (3은 실제로 약 30,000달러를 의미함) 

2. housing median age, median house value 양 끝을 보면 min, max를 한정지었음. housing_median_age는 label로 사용된다(머신러닝 학습과정에서 주의 필요)

3. 특성마다 scale이 매우 다르다. Feature Scaling 필요.

4. 많은 그래프들이 right-skewed. 이를 bell-shaped로 바꾸어야 머신러닝 알고리즘에서 패턴 찾기가 수월해진다.

5. 데이터를 깊게 살피기 전, 테스트 세트를 떼어 놓자!
# # 1. 무작위 샘플링

# In[1]:


import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[53]:


train_set, test_set = split_train_test(housing, 0.2)
len(train_set), len(test_set)


# In[54]:


# 데이터셋 업데이트 이후에도 안정적인 훈련셋/테스트셋 분할을 위한 함수 구현
from zlib import crc32


# In[56]:


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier))&0xffffffff < test_ratio*2**32


# In[62]:


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[64]:


#housing data 내에는 식별자 컬럼이 없음. 대신 행 인덱스를 id로 사용.
housing_with_id = housing.reset_index()


# In[65]:


train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')


# In[66]:


#또는 새로운 식별자를 생성할 수도 있음. 대신 이 식별자는 고유식별자여야 하고, 안정적이어야 함. 
## 구역의 위도와 경도를 사용해 식별자컬럼 생성

housing_with_id['id'] = housing['longitude']*1000 + housing['latitude']
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')


# In[68]:


#사이킷런 : 데이터셋을 여러 서브셋으로 나눌 수 있도록 해준다.

from sklearn.model_selection import train_test_split
# 사이킷런 내의 train_test_split 함수 : 
## 1. 난수의 초깃값을 지정할 수 있는 random_state 매개변수를 가진다 / 여기서는 42로 지정함.
## 2. 행갯수가 같은 데이터셋을 넘겨서 같은 인덱스 기반으로 나눌 수 있다.

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)


# # 2. 계층적 샘플링

# In[76]:


# 가정 : median income이 주택 가격 예측에 매우 중요하다는 정보. 즉 테스트 세트는 소득 카테고리를 잘 대표해야 한다.
# 소득에 대한 카테고리 특성을 만들어야 한다. 소득에 대해 더 자세히 살펴보자.

plt.hist(housing.median_income)

x축을 보자. 대부분 1.5~6 사이에 모여 있다. 계층별로 충분한 데이터셋이 있어야 하고, 너무 많은 계층으로 나누면 안된다.
# In[77]:




# 소득 카테고리 특성을 만드는 함수 pd.cut을 사용해 income_cat이라는 특성을 housing 데이터셋에 추가하자.
housing['income_cat'] = pd.cut(housing['median_income'], 
                               bins=[0.,1.5,3.0,4.5,6.,np.inf],
                               labels=[1,2,3,4,5])


# In[78]:


housing.info()


# In[79]:


#마지막에 income_cat 함수가 category 즉 범주형으로 추가된 것을 볼 수 있다.
housing.income_cat


# In[80]:


#5개의 카테고리로 분류했다. 카테고리 1은 0부터 1.5까지를 의미한다.이제 이를 다시 히스토그램으로 나타내 보자.
housing['income_cat'].hist()


# In[83]:


# 조금 더 보기 깔끔해졌다. 소득 카테고리를 기반으로 계층 샘플링 할 준비 완료! 
# 사이킷런의 Stratified Shufflesplit를 사용해보자.

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[84]:


# 계층 샘플링이 완료되었다. 
#테스트 세트에서 소득 카테고리 비율을 먼저 살핀 후, 훈련 세트에서 소득 카테고리 비율을 확인해보자.

## 테스트 세트
strat_test_set['income_cat'].value_counts()/len(strat_test_set)


# In[86]:


##훈련 세트
strat_train_set['income_cat'].value_counts()/len(strat_train_set)


# In[87]:


#비슷하게 나오는 것을 볼 수 있다 - 사실 거의 똑같다. 

#income_cat 특성을 삭제하고 테스트셋을 원래대로 되돌리자.
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True) #axis = 1은 열 삭제. inplace=True이면 Dataframe 자체를 수정하고 반환하지 X.


# # 3. 데이터 탐색과 시각화

# In[88]:


housing = strat_train_set.copy()


# In[89]:


# 1) 위도, 경도 산점도
housing.plot(kind='scatter', x='longitude', y='latitude')


# In[90]:


# 캘리포니아 지역의 모양을 잘 보여준다. 그러나 일정한 패턴을 찾기는 힘들다. 점을 다르게 표현해보자.
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)


# In[91]:


# alpha 옵션을 0.1로 줘 보았다. 밀집된 영역이 진하게 나타나는 것을 볼 수 있다. 

# 2) 주택 가격 산점도
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, 
            s=housing['population']/100, label='population', figsize=(10,7),
            c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

plt.legend()

빨간색이 높은 가격, 파란색이 낮은 가격을 보여준다. color를 median_house_value로 설정한 것 확인!
# In[93]:


# 상관관계를 조사하고 상관계수를 알아보자. corr() 매서드 활용

corr_matrix = housing.corr()


# In[94]:


corr_matrix['median_house_value'].sort_values(ascending=False) # 상관계수가 큰 순서대로 정리


# In[95]:


# median_income과 관련이 큰 것을 볼 수 있다. 이렇게 숫자로 봐도 되고, 또는 산점도로 확인해도 괜찮다.
## pandas 의 scatter_matrix 함수 사용해서 몇개만 보자.

from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8))


# In[96]:


# 대각선 방향에 그려진 그림은 '각 특성의 히스토그램'이다 자기 자신과의 상관계수는 그냥 직선이라 그려봤자 유용하지 않기 때문.
## median_income이 가장 유용하니까 확대해보자.

housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)


# In[97]:


# 상관관계가 정말정말 강하다. 
# 맨 위에 그려진 진한 상한선이나, 300000~400000 사이에 직선이 보인다. 알고리즘이 이러한 형태를 학습하면 안됨 -> 제거해줘야 함.(이상한 데이터,,)


# 특성 조합을 만들어보자.
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']


# In[99]:


corr_matrix = housing.corr() # 상관관계 행렬을 다시 보자
corr_matrix['median_house_value'].sort_values(ascending=False)

bedrooms_per_room 특성이 total_bedrooms, total_rooms 보다 더 상관관계가 높다. 
interpret : 
 침대, 방 비율이 낮은 집이 더 비싼 경향이 있다. 
 가구당 방 개수가 구역 내 total 방 개수보다 더 유용하다. 
 당연하지만, 큰 집일수록 더 비싸다. 
# # 4. 머신러닝 알고리즘을 위한 데이터 준비 

# In[2]:


housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set.median_house_value.copy()


# In[ ]:




