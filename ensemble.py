import os
import pandas as pd

if __name__ == "__main__":
  import inference

  model_path = './model/'

  pts=os.listdir(model_path)

  labels = []
  for pt in pts:
    #pts는 model 하위에 있는 모델이름 폴더 리스트
    labels.extend(inference(model_name=pt.replace('%','/'),))
    
    label, l=[], len(labels)
  for i in zip(*labels):
    label.append(sum(i)/l)

  output = pd.read_csv(output_sample_path='./data/sample_submission.csv')
  output['label'] = label
  output.to_csv(f'output.csv', index=False)