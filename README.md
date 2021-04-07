# Baseline Model
## Usage
+ Get pretrained Glove word embedding
```
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```



+ Get bert embedding for all tags.
```
python get_bert_embedding.py
```
It will generate two files that will be further used in the below pipeline. \
See more descriptions in the py file.\
Haven't added arguments for setting path and name, should change it manually.

+ Generate adv examples
```
python gen_adv_pipline.py --pre_emb ../glove_emb/glove.6B.100d.txt --save_dir ../para_text/ --name 5_train --dataset train --order rep --n 5
```
This command will ouptut a file named '5_train_rep'.


+ Train
```
python train.py --char_mode LSTM --char_dim 30 --char_lstm_dim 50 --pre_emb ../glove_emb/glove.6B.100d.txt --name rep --non_gradient --per_adv 5 --adv_path ../para_text/5_train_rep
```

+ Get inference metrics
```
python adv_example_eval.py --char_mode LSTM --char_dim 30 --char_lstm_dim 50 --pre_emb ../glove_emb/glove.6B.100d.txt --reload 1 --name normal --per_adv 10 --adv_path ../para_text/10_dev_new_dev_rep
```
