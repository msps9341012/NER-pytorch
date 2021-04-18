# Baseline Model
## Usage
+ Get pretrained Glove word embedding
```
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```



+ Get bert embedding for all tags.
```
python get_bert_embedding.py dataset pooling_method
```
It will generate two files that will be further used in the generating pipeline.
- Options:
    - pooling_method: first, max, mean
    - dataset: train, dev

+ Generate adv examples
```
python gen_adv_pipline.py --pre_emb ../glove_emb/glove.6B.100d.txt --save_dir ../para_text/ --name 5_train --dataset train --order rep --rep_with farthest --n 5 --filter 
```
This command will ouptut a file named '5_train_rep', which applies the perplexity filter and replace with farthest embedding. \
If using bert embedding, add '--bert' and also '--bert_pooler pooling_method' (the one that you used in the above command.) \
As for '--order', you can set something like '--order rep,ppdb,para' to generate sequentially. \
If you want to used the preprocessed data that stored in your local space, see the description of '--preprocess_set'.


+ Train
```
python train.py --char_mode LSTM --char_dim 30 --char_lstm_dim 50 --pre_emb ../glove_emb/glove.6B.100d.txt --name rep --non_gradient --per_adv 5 --adv_path ../para_text/5_train_rep
```

+ Get inference metrics
```
python adv_example_eval.py --char_mode LSTM --char_dim 30 --char_lstm_dim 50 --pre_emb ../glove_emb/glove.6B.100d.txt --reload baseline --per_adv 10 --adv_path ../para_text/10_dev_new_dev_rep
```
