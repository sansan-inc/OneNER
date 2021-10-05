# OneNER

A BERT-based Named Entity Recognition.

## Usage

### Install

```
git clone git@github.com:sansan-inc/OneNER.git
cd OneNER
pip install .
```

### Training

OneNER provides BERT model with a CRF layer or a linear layer for NER tasks.
There are example codes.

Training script requires dataset(train.taw.txt, test.raw.txt, dev.raw.txt).

**Dataset Format**

```txt
Sans B-ORGANIZATION
an I-ORGANIZATION
株 I-ORGANIZATION
式 I-ORGANIZATION
会 I-ORGANIZATION
社 I-ORGANIZATION
が O
提 O
供 O
。 O

名 O
刺 O
管 O
理 O
サービス O
で O
す O
。 O
```


```sh
$ cd examples
$ ls crf_dataset
dev.raw.txt test.raw.txt train.raw.txt

$ bash train_crf.sh
```

### Prediction

```sh
>>> from onener.transformer.ner import BertNERTagger
>>> tagger = BertNERTagger("crf_output")
>>> tagger.predict("Sansan株式会社が提供する名刺管理サービスです。")
[[{'word': 'Sans', 'score': 0.9207938313484192, 'entity': 'B-ORGANIZATION', 'index': 1}, {'word': 'an', 'score': 0.9845695495605469, 'entity': 'I-ORGANIZATION', 'index': 2}, {'word': '株', 'score': 0.9861224889755249, 'entity': 'I-ORGANIZATION', 'index': 3}, {'word': '式', 'score': 0.9906498789787292, 'entity': 'I-ORGANIZATION', 'index': 4}, {'word': '会', 'score': 0.9887092113494873, 'entity': 'I-ORGANIZATION', 'index': 5}, {'word': '社', 'score': 0.9893214106559753, 'entity': 'I-ORGANIZATION', 'index': 6}, {'word': 'が', 'score': 0.9982619285583496, 'entity': 'O', 'index': 7}, {'word': '提', 'score': 0.9966153502464294, 'entity': 'O', 'index': 8}, {'word': '供', 'score': 0.9975463151931763, 'entity': 'O', 'index': 9}, {'word': 'する', 'score': 0.9980711340904236, 'entity': 'O', 'index': 10}, {'word': '名', 'score': 0.9884718060493469, 'entity': 'O', 'index': 11}, {'word': '刺', 'score': 0.9883087873458862, 'entity': 'O', 'index': 12}, {'word': '管', 'score': 0.9875956177711487, 'entity': 'O', 'index': 13}, {'word': '理', 'score': 0.9814469218254089, 'entity': 'O', 'index': 14}, {'word': 'サービス', 'score': 0.9934806227684021, 'entity': 'O', 'index': 15}, {'word': 'で', 'score': 0.9978121519088745, 'entity': 'O', 'index': 16}, {'word': 'す', 'score': 0.9976488351821899, 'entity': 'O', 'index': 17}, {'word': '。', 'score': 0.9979578256607056, 'entity': 'O', 'index': 18}]]
```