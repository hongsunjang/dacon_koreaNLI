import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH =  './'

train = pd.read_csv(os.path.join(PATH,'data/benchmark_train_data.csv'), encoding='utf-8')
test = pd.read_csv(os.path.join(PATH, 'data/test_data.csv'), encoding='utf-8')

train.head(5)

print(train.info(), end='\n\n')
print(test.info())

print('Train Columns: ', train.columns)
print('Test Columns: ', test.columns)

print('Train Label: ', train['label'].value_counts(), sep='\n', end='\n\n')
print('Test Label: ', test['label'].value_counts(), sep='\n')

print('Train Null: ', train.isnull().sum(), sep='\n', end='\n\n')
print('Test Null: ', test.isnull().sum(), sep='\n')

feature = train['label']

plt.figure(figsize=(10,7.5))
plt.title('Label Count', fontsize=20)

temp = feature.value_counts()
plt.bar(temp.keys(), temp.values, width=0.5, color='b', alpha=0.5)
plt.text(-0.05, temp.values[0]+20, s=temp.values[0])
plt.text(0.95, temp.values[1]+20, s=temp.values[1])
plt.text(1.95, temp.values[2]+20, s=temp.values[2])

plt.xticks(temp.keys(), fontsize=12) # x축 값, 폰트 크기 설정
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 레이아웃 설정
plt.show() # 그래프 나타내기

max_len = np.max(train['premise'].str.len())
min_len = np.min(train['premise'].str.len())
mean_len = np.mean(train['premise'].str.len())

print('Max Premise Length: ', max_len)
print('Min Premise Length: ', min_len)
print('Mean Premise Lenght: ', mean_len, '\n')

max_len = np.max(train['hypothesis'].str.len())
min_len = np.min(train['hypothesis'].str.len())
mean_len = np.mean(train['hypothesis'].str.len())

print('Max Hypothesis Length: ', max_len)
print('Min Hypothesis Length: ', min_len)
print('Mean Hypothesis Lenght: ', mean_len)

from collections import Counter

plt.figure(figsize=(10,7.5))
plt.title('Premise Length', fontsize=20)

plt.hist(train['premise'].str.len(), alpha=0.5, color='orange')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 레이아웃 설정

plt.show()

train['premise'] = train['premise'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]', '')
test['premise'] = test['premise'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]", "")
train.head(5)

train['hypothesis'] = train['hypothesis'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]', '')
test['hypothesis'] = test['hypothesis'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]", "")
train.head(5)

!pip install transformers

import os
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from transformers import TFBertForMaskedLM

def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

from sklearn.model_selection import StratifiedKFold
CV= 5
skf = StratifiedKFold(n_splits = CV, shuffle=True, random_state=42)
folds=[]

for train_idx,val_idx in skf.split(train['index'], train['label']):
    folds.append((train_idx,val_idx))


class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, label):
        self.pair_dataset = pair_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['label'] = torch.tensor(self.label[idx])
        
        return item

    def __len__(self):
        return len(self.label)

def label_to_num(label):
    label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2, "answer": 3}
    num_label = []

    for v in label:
        num_label.append(label_dict[v])
    
    return num_label

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'accuracy': acc,
  }

# BERT config setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BERT_TYPE = 'klue/roberta-large'
MODEL_NAME = BERT_TYPE
config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = 3

training_args = TrainingArguments(
    output_dir=os.path.join(PATH,'results'),
    num_train_epochs=10,
    per_device_train_batch_size=16,
    save_total_limit=2,
    save_steps=500,
    evaluation_strategy='steps',
    eval_steps = 100,
    load_best_model_at_end = True,
)

best_models = []
fold = 0
print('===============',fold+1,'fold start===============') 
# Get new model from fold
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)    
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.to(device)

train_idx = folds[fold][0]
valid_idx = folds[fold][1]

train_data = train.loc[train_idx]
val_data = train.loc[valid_idx] 
train_label = label_to_num(train_data['label'].values)
val_label = label_to_num(val_data['label'].values)
    
tokenized_train = tokenizer(
      list(train_data['premise']),
      list(train_data['hypothesis']),
      return_tensors="pt",
      max_length=256, # Max_Length = 190
      padding=True,
      truncation=True,
      add_special_tokens=True
    )
tokenized_val = tokenizer(
      list(val_data['premise']),
      list(val_data['hypothesis']),
      return_tensors="pt",
      max_length=256, # Max_Length = 190
      padding=True,
      truncation=True,
      add_special_tokens=True
    )


train_dataset = BERTDataset(tokenized_train, train_label)
val_dataset = BERTDataset(tokenized_val, val_label)

trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
    )



trainer.train()
