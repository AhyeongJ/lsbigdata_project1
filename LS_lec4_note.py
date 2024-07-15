# soft copy
a = [1, 2, 3]
a

b = a
b

a[1] = 4
a
b  # a에 변형된 부분 같이 변형됨. 

# deep cop
a = [1, 2, 3]
a

b = a[:]
b = a.copy() #두가지 방법

a[1] = 4
a
b


## 숫자함수
import math
x = 4
math.sqrt(x) # 제곱근
math.exp(x) # e^x
math.log(10, 10) # 10의 밑 10 로그 값
math.factoria(4) # 팩토리얼 
math.sin(math.radians(90)) # 90도를 라디안으로 사인 함수 계산
math.cos(math.radians(180)) # 180도를 라디안으로 코사인 함수 계산
math.tan(math.radians(45)) # 45도를 라디안으로 탄젠트 함수 계산

# 정규분포 확률밀도 함수 수식 만들기 
def normal(x, mu, sigma):
    factor = math.sqrt(2*math.pi) ** -1
    return factor * math.exp(-0.5 * ((x-mu) / sigma)**2)

normal( 1, 0, 1)

# 복잡한 수식 계산
def my_f(x, y, z):
    return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)
my_f(2, 9, math.pi / 2)  

# snippet 등록하기 
# Tools - Edit Code Snippets 

#NumPy
# Ctrl + Shif + c: 커맨드 처리 
!pip install numpy
import numpy as np
import pandas as pd

a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)
type(a)
a[1:4]

# 빈 배열 생성
x = np.empty(3)
x[0] = 1
x[1] = 4
x[2] = 10
x #float 생성 
x[2]

# np.arange() 함수 
np.array([1, 2, 3, 4, 5])
np.arange(6)  # 0부터 1씩 건너뛰는 숫자 6개 
np.arange(1, 6, 0.5) #1부터 6미만까지 0.5 간격으로
np.arange(0, -100, -1)
-np.arange(0, 100)

# np.linspace(start, stop, num, endpoint = T, retstep = F, dtype =None)
np.linspace(0, 1, 5) #0부터 1까지 5개의 숫자, 균일한 간격 숫자 배열 형성
np.linspace(0, 1, 5, endpoint = False) # 끝지점 포함 안 하기 

# np.repeat(a, repeats, axis = None)
# a: 반복할 입력 배열, repeats: 반복 횟수, axis: 반복 적용 축
np.repeat(3,5)
np.repeat([1, 2, 3], 2)  # [1, 1, 2, 2, 3, 3]
np.repeat([1, 2, 3], repeats = [1, 2, 3])

# np.tile: 벡터 전체 반복 
np.tile([1, 2, 3], 2)

# 벡터끼리 계산
vec1 = np.arange(5)
vec1 + vec1
max(vec1), min(vec1)

# 35672 이하 홀수들의 합은?
x = np.arange(1, 35673, 2)
x.sum()
len(x)
x.shape
x.size

b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) # 첫 번째 차원의 길이 2 
shape = b.shape # 각 차원의 크기 (2, 3)
size = b.size # 전체 요소의 개수 6
pd.DataFrame(b)

# 벡터 연산 
import numpy as np
a = np.array([1, 2])
b = np.array([1, 2, 3, 4])
np.tile(a, 2) + b # 길이가 맞아야 계산 가능
np.tile(a, 2) - b

b == 3   # b의 각각의 원소에 대해 true, false 가능 

# 35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 정수의 갯수 
((np.arange(1, 35673) % 7) == 3).sum()
# False는 0이고 True는 1이므로 sum으로 구할 수 있음. 

# 벡터 계산 하기 
a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
[10.0, 10.0, 10.0],
[20.0, 20.0, 20.0],
[30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])    
vector.shape   # 튜플인데 숫자 하나일 때는 shpae(#,)
vector = vector.reshape(1, 3)
vector.shape

# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

# 세로 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector.shape
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)
