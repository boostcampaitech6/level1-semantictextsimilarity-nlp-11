import os
import pandas as pd

if __name__ == "__main__":
  '''
  pselabel 붙인 csv 파일이 2x파일이므로 절반으로 나눠서 다 평균내면 
  sep로 앞뒤 바뀐 문장들도 같은 모델로  예측한뒤 softvoting하는 효과가 남

  이를 위해 3trance2x nolabel을 사용해서 csv파일에 pselabel 붙인 파일들이
  csv 파일에 있다고 가정
  나눠서 softvoting
  다른 3trance2x pselabel

  '''
  csv_path = './csv/'

  pts = os.listdir(csv_path)

  labels = []
  for csv in pts:
    try:
      df = pd.read_csv(csv_path+csv)
      label=list(df['label'])
      num = len(label)//2
      labels.append(label[:num])
      labels.append(label[num:])
    except:
      pass
  label, l = [], len(labels)
  
  for i in zip(*labels):
    label.append(sum(i)/l)
  
  output = pd.read_csv('./data/3trance2xnolabel.csv')
  output['label'] = label*2
  
  train = pd.read_csv('./data/train2x.csv')
  output = pd.concat([output,train])

  output.to_csv(f'./data/train_deep.csv', index=False)
