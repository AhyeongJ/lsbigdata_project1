import numpy as np

# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024)
a = np.random.randint(1, 21, 10)   # 1에서부터 20까지의 정수 10개 랜덤 추출
print(a)

print(a[1])
print(a[2:5])
print[::2] # 처음부터 끝까지 2씩 건너띄워서 추출
print(a[-2])
print(a[1:6:2])

# 1에서부터 1000사이 3의 배수의 합은?
x = np.arange(3,1001)
x[::3].sum()

print(a[[0, 2, 4]]) #첫번째, 3번째, 5번째 값 추출
np.delete(a,3)
np.delete(a, [1,3])

a > 3 
a[a > 3]   # a를 논리형 벡터로 만들고, true에 해당하는 원소만 추출 

np.random.seed(1020)
a = np.random.randint(1, 10000, 5) 
a
a < 5000
a > 2000
b = a[(a < 5000) & (a > 2000)]
b

!pip install pydataset
import pydataset
df = pydataset.data('mtcars')
np_df = np.array(df['mpg'])   # 숫자만, 1차원 배열로 뽑아옴 
df['mpg'][df['mpg'] >2000]
model_names = np.array(df.index)

# 15 이상이고 20이하인 데이터 개수는 
sum((np_df >= 15) & (np_df <= 20))
model_names[(np_df >= 15) & (np_df <= 20)] #자동차 모델명

# 15 보다 작거나 22 이상인 데이터 개수
sum((np_df < 15) | (np_df >=22))

# 평균 mpg 이상인 자동차 대수 
sum(np_df >= np.mean(np_df))
model_names[np_df >= np.mean(np_df)]  # 자동차 모델명 

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])
# a[조건을 만족하는 논리형벡터]
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)]

a[a > 3000] = 3000
a

np.random.seed(2024)
a = np.random.randint(1, 100, 10)
a < 50
np.where(a < 50)   #True의 위치 반환 

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a
# 처음으로 22000보다 큰 숫자 나왔을 때, 
# 숫자 위치와 그 숫자는 무엇인가요?
x = np.where(a > 22000) 
type(x)   # 튜플
x[0]  # numpy array 
my_index = x[0][0]  # 숫자의 위치: 10 (11번째) 
a[my_index]  # 숫자: 24651


# 처음으로 24000보다 큰 숫자가 나왔을 때
# 숫자 위치와 그 숫자는?
x = np.where(a > 24000)
my_index = x[0][0]
a[my_index]

# 10000 보다 큰 숫자 중에서 
# 50번째로 나온 숫자 위치와 그 숫자는?
x = np.where(a >10000)  # 얘는 조건에 해당하는 위치를 뽑아냄
my_index = x[0][49]
my_index
a[my_index]

# 아래와 같이 할 경우 숫자의 위치를 파악할 수 없음. 
x = a[a>10000]   # 얘는 조건에 해당하는 숫자를 뽑아냄
x
x[49]

# 500보다 작은 숫자들 중
# 가장 마지막으로 나오는 숫자 위치와 그 숫자는?
x = np.where(a < 500)
index = x[0][-1]  # 961번째 위치 
a[index]   # 숫자: 391 


# np.nan (not a number)
a = np.array([20, np.nan, 13])
import pandas as pd
df_a = pd.DataFrame(a)
print(df_a)
df_a.mean()
    a
a + 3 
np.mean(a)  
np.nanmean(a) #nan 무시 함수

# None은 값이 없음을 나타내는 특수 상수 
a = None
b = np.nan
b
a
b + 1 
a + 1 # error

# 빈 칸 제거하는 방법
a = np.array([20, np.nan, 13])
np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered  # nan 값 빼고 가져옴. 

# 벡터 합치기 
str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0,2]]

# 리스트: 데이터타입 두개 이상 가능
# 벡터: 데이터 타입 하나만 
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype =str) # 통일할 수 있는 타입 정보 입력 
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec))  # 튜플로 묶음 소괄호 사용
combined_vec
combined_vec = np.concatenate([str_vec, mix_vec])  # 리스트로 묶음
combined_vec

col_stacked = np.column_stack((np.arange(1,5), 
                               np.arange(12,16)))
col_stacked

row_stacked = np.vstack((np.arange(1, 5),
                            np.arange(12, 16)))
row_stacked


vec1 = np.arange(1,5)
vec2 = np.arange(12,18)

vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked= np.column_stack((vec1, vec2))  
uneven_stacked

# 연습문제 
# 1. 
a = np.array([12, 21, 35, 48, 5])

#2.  주어진 벡터의 홀수 번째 요소만 추출하여 새로운 벡터 생성
a[0::2]   

# 3. 최댓값 찾기
np.max(a)

# 4. 중복된 값을 제거한 새로운 벡터를 생성
np.unique(a)

# 5. 주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성하라
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])

x = np.empty(6)
x[0::2] = a
x[1::2] = b

x[[0,2,4]] = a
x[[1,3,5]] = b
x

# 6.다음 a 벡터의 마지막 값은 제외한 두 벡터 a와 b를 더한 결과를 구하라
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])
a[:-1] + b

