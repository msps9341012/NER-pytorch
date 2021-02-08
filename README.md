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

## To Do
+ Paper 1
   - [ ] Layer Normalization
   - [ ] Implement gradient estimation

+ Paper 2
   - [ ] Combine ELMo embeddings
   - [ ] Add masking function

## Performance
F1 score on CoNLL-2003 English task.

| Method        | Paper | Our |
|---------------|-------|-----|
| BiLSTM-CRF    | 91.22 |     |
| BiLSTM-CRF+AT | 91.56 |     |
| CNN-LSTM-CRF  |       |     |
| MAT           |       |     |
| ....          |       |     |

## Paper References
1. [Robust Multilingual Part-of-Speech Tagging via Adversarial Training](https://www.aclweb.org/anthology/N18-1089.pdf)
   - [Code](https://github.com/michiyasunaga/pos_adv)
2. [Enhance Robustness of Sequence Labelling with Masked Adversarial Training](https://www.aclweb.org/anthology/2020.findings-emnlp.28/)
3. [Neural Architectures for Named Entity Recognition End-toEnd Sequence labeling via BLSTM-CNN-CRF](https://arxiv.org/pdf/1603.01354.pdf)
   - [Code](https://github.com/glample/tagger)
---
# Possible ideas
Papers that we can look into
1. [A Retrieve-and-Edit Framework for Predicting Structured Outputs](https://papers.nips.cc/paper/2018/file/cd17d3ce3b64f227987cd92cd701cc58-Paper.pdf)

