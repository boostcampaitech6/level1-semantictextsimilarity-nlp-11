"""
sentence_1 및 sentence_2 열에서  특수문자 제거 + 초성 제거 + 빈 값 제거
"""
import pandas as pd
import re

def clean_data(input_path, output_path='./cleaned_data.csv'):
    # dataframe 생성
    df = pd.read_csv(input_path)

    # 특수문자 제거
    df['sentence_1'] = df['sentence_1'].apply(lambda row : re.sub(r'[^a-zA-Z0-9가-힣ㄱ-ㅎ\s]', ' ', str(row)))
    df['sentence_2'] = df['sentence_2'].apply(lambda row : re.sub(r'[^a-zA-Z0-9가-힣ㄱ-ㅎ\s]', ' ', str(row)))

    # 연속적으로 반복하는 초성을 2개로 축소
    df['sentence_1'] = df['sentence_1'].apply(lambda row : re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1+', r'\1'*2, row))
    df['sentence_2'] = df['sentence_2'].apply(lambda row : re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1+', r'\1'*2, row))


    # NaN 제거
    df = df.dropna(axis=0)

    # 중복 행 제거
    df = df.drop_duplicates(['sentence_1', 'sentence_2'], keep='first')

    # csv 파일 생성
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = './data/train.csv'
    output_path = './cleaned_data.csv'
    clean_data(input_path, output_path)