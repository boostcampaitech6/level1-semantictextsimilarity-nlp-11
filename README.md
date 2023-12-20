
./data/3trance2xnolabel.csv
./data/train2x.csv

data폴더에 3trance2xnolabel.csv,train2x.csv 추가
infetrance.py에 model_name 변경

python inftrans.py 실행

./csv/{model_name}_pseudolabel.csv 파일 생성


python csvtransfer.py 실행

csv 폴더에 있는 모든 csv파일을 읽어서 Ensemble된 csv파일에 train2x.csv를 추가하여
./data/train_deep.csv 파일을 생성함

이 train_deep.csv 파일을 사용하여 train.py에 train_path에 추가하여 작업함

