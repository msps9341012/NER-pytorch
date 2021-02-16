# Baseline Model
## Usage
+ Get pretrained Glove word embedding
```
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```
+ Train and evaluate.
   + It supports CNN and Bi-LSTM for the character level.
   + The below example use the parameters settings from [Robust Multilingual Part-of-Speech Tagging via Adversarial Training](https://www.aclweb.org/anthology/N18-1089.pdf)
```
cd NER-pytorch
!python train.py --char_mode LSTM --char_dim 30 --char_lstm_dim 50 --pre_emb ../glove.6B.100d.txt
```
+ To see all supporting arguments,
```
!python train.py -h
```

