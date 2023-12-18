from model import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import random
import datetime
import get_model_path as get

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='klue/roberta-small', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_epoch', default=100, type=int)
parser.add_argument('--shuffle', default=True)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--train_path', default='./data/train.csv')
parser.add_argument('--dev_path', default='./data/dev.csv')
parser.add_argument('--test_path', default='./data/dev.csv')
parser.add_argument('--predict_path', default='./data/test.csv')
args = parser.parse_args(args=[])

dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                        args.test_path, args.predict_path)
model = Model(args.model_name, args.learning_rate)


# Early Stoping
now = datetime.datetime.now()
timestamp = now.strftime("%d%H%M%S")

# checkpoint = ModelCheckpoint(monitor="val_pearson", save_top_k=1, dirpath='./model', filename=f"{get.get_safe_filename(args.model_name)}_"+"{epoch}-{val_pearson:.2f}"+timestamp, verbose=False, mode="min")
checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, dirpath='./model/'+get.get_safe_filename(args.model_name), filename='{epoch}-{val_pearson:.4f}_'+f'{timestamp}', verbose=False, mode="min")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")

# gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
trainer = Trainer(accelerator="auto", devices=1, max_epochs=args.max_epoch, log_every_n_steps=100, callbacks=[early_stop_callback,checkpoint])

# Train part
trainer.fit(model=model, datamodule=dataloader)
trainer.test(model=model, datamodule=dataloader)

# 학습이 완료된 모델을 저장합니다.

# torch.save(model, '')
