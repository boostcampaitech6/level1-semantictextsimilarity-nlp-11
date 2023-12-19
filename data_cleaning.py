#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from collections import Counter

from hanspell import spell_checker

path = '../data/train.csv'
df = pd.read_csv(path)

### 이 파일에 있는 셀들 다 실행하시면 됩니다.
### new_cleaned_data.csv가 생성될텐데 그걸로 훈련돌리면 됩니다.

# %%

################# (특수문자 제거 + 초성 제거 + 빈 값 제거) sequence ###############

new_path='../data/3trans2x_train2x.csv'

# sentence_1 및 sentence_2 열에서 특수문자 제거
def remove_special_characters(text):
    # 정규 표현식을 사용하여 특수문자 제거
    return re.sub(r'[^a-zA-Z0-9가-힣ㄱ-ㅎ\s]', '', str(text))

# 초성만 있는 문자를 제거하는 함수
def remove_initial_consonant(text):
    # 초성 정규표현식
    initial_vowel_pattern = re.compile('[ㄱ-ㅎㅏ-ㅣ]+')
    
    # 정규표현식을 사용하여 초성 제거
    result = re.sub(initial_vowel_pattern, '', text)
    return result

# nan 제거
def remove_nan(df):
    df = df.dropna()
    return df

# csv 파일 생성
def make_csv(df):
    df.to_csv('new_cleaned_train_data.csv', index=False)


def clean_data(path):
    # dataframe 생성
    df = pd.read_csv(path)

    # 'sentence_1' 및 'sentence_2' 열에 함수 적용하여 특수문자 제거
    df['sentence_1'] = df['sentence_1'].apply(remove_special_characters)
    df['sentence_2'] = df['sentence_2'].apply(remove_special_characters)

    # 초성만 있는 문자를 제거하여 새로운 열 추가
    df['sentence_1'] = df['sentence_1'].apply(remove_initial_consonant)
    df['sentence_2'] = df['sentence_2'].apply(remove_initial_consonant)

    # NaN 제거
    df = df[df['sentence_1'] != np.nan]
    df = df[df['sentence_2'] != np.nan]
    df = df.dropna()

    return df
    

df = clean_data(new_path)

df = df[df['sentence_1'] != np.nan]
df = df[df['sentence_2'] != np.nan]
df = df.dropna()

df.to_csv('new_cleaned_train_data.csv', index=False)


#%%

new_path='../code/new_cleaned_train_data.csv'
df = pd.read_csv(new_path)

df = df[df['sentence_1'] != np.nan]
df = df[df['sentence_2'] != np.nan]
df = df.dropna()

df.to_csv('new_cleaned_data.csv', index=False)



