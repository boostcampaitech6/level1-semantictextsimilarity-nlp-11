from model import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import random
import datetime
import numpy as np
import get_model_path as get

output_sample_path_main=''
model_name_main=''

def inference(
    model_name="klue/roberta-small",
    train_path="./data/train.csv",
    dev_path="./data/dev.csv",
    test_path="./data/dev.csv",
    predict_path="./data/3trance2xnolabel.csv",
    batch_size=16,
    shuffle=True,
    learning_rate=1e-5,
    max_epoch=1,
    output_sample_path='./data/3trance2xnolabel.csv'
):
  """
  모델을 불러와서 예측을 수행하고 결과를 저장하는 함수입니다.

  Args:
      model_name: 모델 이름 (예: klue/roberta-small)
      train_path: 학습 데이터 경로
      dev_path: 개발 데이터 경로
      test_path: 테스트 데이터 경로
      predict_path: 예측 결과 저장 경로
      batch_size: 배치 크기 (default: 16)
      shuffle: 데이터 셔플 여부 (default: True)
      learning_rate: 학습률 (default: 1e-5)
      max_epoch: 최대 에포크 수 (default: 1)
  """

  # parser를 사용하지 않고도 inference() 함수를 사용할 수 있습니다.

  # dataloader와 model을 생성합니다.
  dataloader = Dataloader(model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path)

  # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
  trainer = Trainer(accelerator="auto", devices=1, max_epochs=max_epoch, log_every_n_steps=1)

  # 모델 불러오기
  bestpt="./model/"+get.get_safe_filename(model_name)+'/'+get.get_best_checkpoint("./model/"+get.get_safe_filename(model_name))
  model = torch.load(bestpt)

  # 예측 수행
  predictions = trainer.predict(model=model, datamodule=dataloader)

  # 결과 처리
  predictions = [(float(i)) for i in torch.cat(predictions)]
  output_sample_path_main=output_sample_path
  model_name_main=model_name

  return predictions


if __name__ == "__main__":
  output = pd.read_csv(output_sample_path_main)
  output['label'] = inference()

  output["label"] = np.where(output["label"] >= 5, 5.0, output["label"])
  output["label"] = np.where(output["label"] <= 0, 0.0, output["label"])
  
  output.to_csv(f'./data/{model_name_main}_pseudolabel.csv', index=False)