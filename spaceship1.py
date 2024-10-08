import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

## 필요한 데이터 불러오기
train = pd.read_csv("C:/Users/USER/Documents/LS 빅데이터 스쿨/lsbigdata_project1/space-titanic/data/train.csv")
test = pd.read_csv("C:/Users/USER/Documents/LS 빅데이터 스쿨/lsbigdata_project1/space-titanic/data/test.csv")
sub_df = pd.read_csv("C:/Users/USER/Documents/LS 빅데이터 스쿨/lsbigdata_project1/space-titanic/data/sample_submission.csv")


## 필요없는 칼럼 제거
all_data=pd.concat([train, test])
all_data=all_data.drop(['PassengerId', 'Name'], axis=1)

# 범주형 칼럼
c = all_data.columns[all_data.dtypes == object][:-1]

# 정수형 전처리
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in c:
    all_data[i] = le.fit_transform(all_data[i])


# 결측치 처리
all_data.fillna(-99, inplace=True)

# train / test 데이터셋
train_n=len(train)
train=all_data.iloc[:train_n,]
test=all_data.iloc[train_n:,]

# 타겟 변수 분리
y = train['Transported'].astype("bool")

# OneHotEncoder 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first',
                              handle_unknown='ignore'), c)
    ], remainder='passthrough')

# 학습 데이터와 테스트 데이터에 전처리 적용
X_train = preprocessor.fit_transform(train.drop(['Transported'], axis=1))
X_test = preprocessor.transform(test)


# 모델 생성 및 학습
base_model2 = RandomForestClassifier(n_estimators= 100,random_state=42)
# 랜덤포레스트의 하이퍼파라미터 그리드
param_grid = {
    'max_depth': [None, 5, 10],  # 트리의 최대 깊이
    'min_samples_split': [8, 10, 15],  # 노드를 나누는 최소 샘플 수
    'max_features': ['sqrt', 'log2']  # 트리에서 고려할 최대 특성 수
}

# 그리드서치 실행
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',  # 또는 'neg_log_loss', 'roc_auc' 같은 다른 평가 지표
    cv=5  # 5-fold 교차 검증
)

grid_search.fit(X_train, y)

# 최적의 하이퍼파라미터와 성능 확인
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
prediction_best_rf = best_rf.predict(X_test)




# Transported 바꾸기
sub_df["Transported"] = prediction_best_rf
sub_df

# csv 파일로 내보내기
sub_df.to_csv("C:/Users/USER/Documents/LS 빅데이터 스쿨/lsbigdata_project1/space-titanic/data/rf.csv", index=False)


