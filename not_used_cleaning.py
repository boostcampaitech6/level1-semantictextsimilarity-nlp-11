#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from collections import Counter

from hanspell import spell_checker

path = '../data/train.csv'
df = pd.read_csv(path)

# 1, 2, 

# %%

########## 이거 사용하시면 됩니다. ##############
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



#%%
################ original data 라벨 개수 시각화 #####################

# train data 시각화
def visualize_train_label_ratio():
    # train data 가져오기
    train_path = '../data/train.csv'
    train_data = pd.read_csv(train_path)
    
    # label counting
    labels = train_data['label']

    print(labels.astype('int').value_counts())

    label_counts = labels.value_counts()
    print(label_counts)

    # 시각화 코드 작성
    plt.figure(figsize=(12,10))
    plt.bar(label_counts.keys(), label_counts.values, width=0.1, color='yellow')
    plt.xticks(label_counts.keys())

    plt.show()

# dev(validation) data 시각화
def visualize_val_label_ratio():
    # validation 데이터 불러오기
    validation_path = '../data/dev.csv'
    validation_data = pd.read_csv(validation_path)

    # count label
    label = validation_data['label']

    print("int 단위 라벨별 count\n", label.astype('int').value_counts())

    label_counts = label.value_counts()
    print("label별 count\n", label_counts)

    # 시각화    
    plt.figure(figsize=(12,10))  
    plt.bar(label_counts.keys(), label_counts.values, width=0.1, color='green')
    plt.xticks(label_counts.keys())

visualize_train_label_ratio()
visualize_val_label_ratio()









# %%
################ 중복 제거 #################

# sentence_1과 sentence_2가 중복되는 row는 없다. 
# 데이터 증강 후 중복되는 코드가 발생한다면 사용

# 중복되는 row를 확인한다.
def check_duplicated(path):
    train_df = pd.read_csv(path)
    duplicated_df = train_df[train_df.duplicated(subset=['sentence_1', 'sentence_2'])]
    print(duplicated_df)

# 중복되는 row를 삭제한다.
def remove_duplicated(path):
    train_df = pd.read_csv(path)

path = '../data/train.csv'
check_duplicated(path)









# %%
########### 특수문자 제거 #############

print("특수문자 제거 전", df[['sentence_1', 'sentence_2']].head(5))

# sentence_1 및 sentence_2 열에서 특수문자 제거
def remove_special_characters(text):
    # 정규 표현식을 사용하여 특수문자 제거
    return re.sub(r'[^a-zA-Z0-9가-힣ㄱ-ㅎ\s]', '', str(text))

# 'sentence_1' 및 'sentence_2' 열에 함수 적용하여 특수문자 제거
df['sentence_1'] = df['sentence_1'].apply(remove_special_characters)
df['sentence_2'] = df['sentence_2'].apply(remove_special_characters)

print("특수문자 제거 후", df[['sentence_1', 'sentence_2']].head(5))









# %%
####### 맞춤법 교정 #############

def correct_spell(text):
    spelled_sentence = spell_checker.check(text)
    correct_sentence = spelled_sentence.checked
    return correct_sentence

df['sentence_1'] = df['sentence_1'].apply(correct_spell)
df['sentence_2'] = df['sentence_2'].apply(correct_spell)







# %%
########### hanspell 라이브러리 정상 동작 여부 테스트 코드 #################3

sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
spelled_sent = spell_checker.check(sent)
## 맞춤법 체크된 문장 할당
hanspell_sent = spelled_sent.checked
print(hanspell_sent)







# %%
######### 초성 제거 (맞춤법 교정하면 자동으로 반영됨) #############

new_path='./fixed_train.csv'
# 초성만 있는 문자를 제거하는 함수
def remove_initial_consonant(text):
    # 초성 정규표현식
    initial_vowel_pattern = re.compile('[ㄱ-ㅎㅏ-ㅣ]+')
    
    # 정규표현식을 사용하여 초성 제거
    result = re.sub(initial_vowel_pattern, '', text)
    return result

# 예제 데이터프레임 생성
df = pd.read_csv(path)
print(df[['sentence_1', 'sentence_2']].head(10))

# 초성만 있는 문자를 제거하여 새로운 열 추가
df['sentence_1'] = df['sentence_1'].apply(remove_initial_consonant)
df['sentence_2'] = df['sentence_2'].apply(remove_initial_consonant)

# 결과 출력
print(df[['sentence_1', 'sentence_2']].head(10))

df.to_csv('fix_completed_train.csv', index=False)






# %%
#################### df를 기반으로 csv 파일 생성 코드 ######################

def make_csv(df):
    df.to_csv('cleaned_grammer_special_data.csv', index=False)

make_csv(df)





# %%
##################### 빈 값 제거 ####################3#

ndf = pd.read_csv('./tmp.csv')

def remove_nan(df):
    return df.dropna(axis=0)

ndf = remove_nan(ndf)
ndf.to_csv('tmp_exp.csv', index=False)




# %%

########## 이거 사용하시면 됩니다. ##############
################# (특수문자 제거 + 초성 제거 + 빈 값 제거) sequence ###############

path=''

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
    return df.dropna(axis=0)

# csv 파일 생성
def make_csv(df):
    df.to_csv('cleaned_train_data.csv', index=False)


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
    df = remove_nan(df)

    # csv 파일 생성
    make_csv(df)


