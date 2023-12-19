from model import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import random
import datetime
import get_model_path as get
import pytorch_lightning as pl
import wandb

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='klue/roberta-small', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_epoch', default=1, type=int)
parser.add_argument('--shuffle', default=True)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--train_path', default='../data/train.csv')
parser.add_argument('--dev_path', default='../data/dev.csv')
parser.add_argument('--test_path', default='../data/dev.csv')
parser.add_argument('--predict_path', default='../data/test.csv')
args = parser.parse_args(args=[])

dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                        args.test_path, args.predict_path)
model = Model(args.model_name, args.learning_rate)


# Early Stoping
now = datetime.datetime.now()
timestamp = now.strftime("%d%H%M%S")

# checkpoint = ModelCheckpoint(monitor="val_pearson", save_top_k=1, dirpath='./model', filename=f"{get.get_safe_filename(args.model_name)}_"+"{epoch}-{val_pearson:.2f}"+timestamp, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                             save_top_k=1, 
                             dirpath='./model/'+get.get_safe_filename(args.model_name), 
                             filename='{epoch}-{val_pearson:.4f}_'+f'{timestamp}', 
                             verbose=False, 
                             mode="min")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")

# gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
trainer = Trainer(accelerator="auto", devices=1, max_epochs=args.max_epoch, log_every_n_steps=100, 
                  callbacks=[early_stop_callback, checkpoint_callback])

# W&B 연동을 위한 설정
wandb.init(project='Level-1', entity='nlp-11_4intcute', config=vars(trainer))
wandb_logger = pl.loggers.WandbLogger()
wandb.run.name = "run_name" # wandb에서
wandb.run.save()
trainer.logger = wandb_logger

# Train part
trainer.fit(model=model, datamodule=dataloader)

# 학습이 완료된 모델을 저장합니다.
best_model_path = checkpoint_callback.best_model_path
best_model = Model.load_from_checkpoint(best_model_path)

# 최적의 metric 값
best_val_loss = early_stop_callback.best_score
print(f'early stop val loss : {best_val_loss}')
wandb.log({"best_val_loss": best_val_loss})

# best pearson 확인
result = trainer.test(model=best_model, datamodule=dataloader)
best_pearson = result[0]['test_pearson']

# model path
epoch_es = early_stop_callback.stopped_epoch
ts = datetime.datetime.now().strftime("%d%H%M%S")
dirpath='./model/'+get.get_safe_filename(args.model_name)
model_name = f"epoch={epoch_es}-val_pearson={best_pearson:.4f}_{ts}.pt"

# model save
torch.save(best_model, os.path.join(dirpath, model_name))
