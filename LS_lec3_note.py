# 데이터 타입
x = 15
print(x, "는 ", type(x), "형식입니다.", sep='') # sep = '' 공백없이 입력
# default: 쉼표는 띄어쓰기 하나로 연결 

## 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

print(a, type(a))
print(b, type(b))


# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(ml_str, type(ml_str))


# 문자열 결합
greeting = "안녕" + " " + "파이썬!"
print("결합된 문자열:", greeting)


#문자열 반복
laugh = "하" * 3
print("반복 문자열:", laugh)


#리스트 
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]

print("Fruits:", fruits)
print("Numbers:", numbers)
print("Mixed List:", mixed_list)

type(fruits)
type(numbers)
## 리스트는 가장 유연함. 다양한 타입의 요소를 포함할 수 있음.


# 튜플
## 리스트와 유사하지만 한 번 생성된 후 수정할 수 없음. 
a_tp = (10, 20, 30)
a_tp
a = 1, 2, 3   # 괄호 안 적어도 튜플로 생성 
a

a_list = [10, 20, 30]

a_tp[1] = 25  # error

a_list[1] = 25
a_list  #변경 됨  


# 단원소 튜플 
c = (42,) # 요소가 하나만 있는 튜플을 만들 때 요소 뒤에 쉼표 붙여야 함. 
c_int = (42)
type(b_int)
c_tp = (42,)
type(b_tp)


#튜플도 인덱싱과 슬라이싱 가능 
b = (1 ,2 ,3 ,4, 5)
b[0]
b[3:]  # 해당 인덱스 이상
b[:3]  # 해당 인덱스 미만 
b[1:3] # 해당 인덱스 이상 & 미만 


# 튜플과 사용자 정의 함수 
def min_max(numbers):
 return min(numbers), max(numbers)  
#괄호가 딱히 지정이 안 되고 콤마로만 연결 되어 있어 튜플로 리턴 


# 딕셔너리
Ahyeong = {
 'name': ('아영', '정'), # 튜플 
 'age': 22,
 'city': ['Seoul', 'Busan']  # 리스트 
}
print("Ahyeong:", Ahyeong)

Ahyeong.get('name')  # 튜플
Ahyeong_name = Ahyeong.get('name')
Ahyeong_name[0]
Ahyeong_city = Ahyeong.get('city') # 리스트 


# 집합
# 중괄호{}를 사용해 생성
# 유니크한 요소만 저장하며, 입력된 순서를 유지하지 않음
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits)  #중복 안 됨. apple 하나만, 순서 상관 안 함. 

# 빈집합
empty_set = set()
print("Empty set:", empty_set)
empty_set.add("apple")  # 한 번에 하나만 추가 가능 
empty_set.add("banana")
empty_set
empty_set.remove("banana")
empty_set.discard("cherry") # error 안 뜸
empty_set.remove("cherry")  # error 뜸

# 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits)
intersection_fruits = fruits.intersection(other_fruits)
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)


# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산됩니다.

is_active = True
is_greater = 10 > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)


# 조건문
a = 3
if (a == 2):
  print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")


# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

#리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

# 논리형과 숫자형 변환 예제
# 숫자를 논리형으로 변환
zero = 0
non_zero = 7
bool_from_zero = bool(zero) # False
bool_from_non_zero = bool(non_zero) # True
print("0를 논리형으로 바꾸면:", bool_from_zero)
print("7를 논리형으로 바꾸면:", bool_from_non_zero)

# 논리형을 숫자로 변환
true_bool = True
false_bool = False
int_from_true = int(true_bool) # 1
int_from_false = int(false_bool) # 0
print("True는 숫자로:", int_from_true)
print("False는 숫자로:", int_from_false)

# 논리형과 문자열형 변환 예제
# 논리형을 문자열로 변환
str_from_true = str(true_bool) # "True"
str_from_false = str(false_bool) # "False"
print("True는 문자열로:", str_from_true)
print("False는 문자열로:", str_from_false)

# 문자열을 논리형으로 변환
str_true = "True"
str_false = "False"
bool_from_str_true = bool(str_true) # True
bool_from_str_false = bool(str_false) # True, 비어있지 않으면 무조건 참
print("'True'는 논리형으로 바꾸면:", bool_from_str_true)
print("'False'는 논리형으로 바꾸면:", bool_from_str_false)


# 교재 63 페이지

# seaborn 패키지 설치 
!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt


var = ['a', 'a', 'b', 'c']
var
seaborn.countplot(x = var, color = "pink")
plt.show()
plt.clf()

df = sns.load_dataset('titanic')
sns.countplot(data = df, x = "sex", hue ="sex")
plt.show()
plt.clf()

sns.countplot(data=df, x="class", hue = "class")
plt.show()
plt.clf()

sns.countplot(data = df, x = 'class', hue = 'alive', orient = "v")
plt.show()
plt.clf()

sns.countplot(data = df, y = 'class', hue = 'alive')  
plt.show()
plt.clf()


!pip install scikit-learn
import sklearn.metrics
from sklearn import metrics 
import sklearn.metrics as met
met.accuracy_score()
