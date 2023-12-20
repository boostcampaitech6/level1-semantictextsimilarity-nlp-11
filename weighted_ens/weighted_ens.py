import os
import pandas as pd

# data 폴더 안에 성능이 좋은 순서대로 이름이 정렬된 csv 파일들이 있어야 합니다. 
# ex) 1등_output.csv, 2등_output.csv, 3등_output.csv
csv_path = './data/'
pts = os.listdir(csv_path)

# DataFrame을 생성하고, id 칼럼과 각 파일들의 예측치 칼럼들을 추가합니다.
df = pd.DataFrame()

for csv in pts:
  try:
    df['id'] = pd.read_csv(csv_path+csv)['id']
    df[csv] = pd.read_csv(csv_path+csv)['target']
  except:
    pass

# Rank에 따른 Weight를 적용합니다. (Ex. 3개의 앙상블 재료 : 1등 * 3, 2등 * 2, 3등 * 1)
columns = list(df.drop(columns = ['id']).columns.values) # id 칼럼 제외 후 연산
n = len(columns)
for i in range(n):
  df[columns[i]] = df[columns[i]].apply(lambda x: x * (n - i))

# 평균 구하기
df['w_sum'] = df.sum(numeric_only=True, axis=1)
df['w_average'] = df['w_sum'].apply(lambda x: x / (n * (n + 1) / 2)) # 위에서 곱한 가중치의 총량을 반영하여 평균을 나눠준다.

# 0보다 작거나, 5보다 큰 값 자르기
df['w_average'] = df['w_average'].apply(lambda x: max(x, 0))
df['w_average'] = df['w_average'].apply(lambda x: min(x, 5))

# 출력될 DataFrame을 생성합니다.
df_out = pd.DataFrame()
df_out['id'] = df['id']
df_out['target'] = df['w_average']

# 생성된 앙상블 결과를 data 폴더 내에 weighted_ensemble.csv 파일로 출력합니다.
df_out.to_csv(csv_path + "weighted_ensemble.csv", index=False)