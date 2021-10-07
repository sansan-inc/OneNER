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
In [1]: from onener.transformer.ner import BertCrfNERTagger

In [2]: tagger = BertCrfNERTagger("examples/crf_output") # set model directory path

In [3]: tagger.predict("Sansan株式会社が提供する名刺管理サービスです。")
Out[3]:
[[{'word': 'Sansan', 'score': 0.91733717918396, 'entity': 'B-組織名', 'index': 1},
  {'word': '株式会社', 'score': 0.9866153001785278, 'entity': 'I-組織名', 'index': 2},
  {'word': 'が', 'score': 0.9982038736343384, 'entity': 'O', 'index': 3},
  {'word': '提供', 'score': 0.9965072274208069, 'entity': 'O', 'index': 4},
  {'word': 'する', 'score': 0.9980195760726929, 'entity': 'O', 'index': 5},
  {'word': '名刺', 'score': 0.9890339374542236, 'entity': 'O', 'index': 6},
  {'word': '管理', 'score': 0.9876256585121155, 'entity': 'O', 'index': 7},
  {'word': 'サービス', 'score': 0.9932954907417297, 'entity': 'O', 'index': 8},
  {'word': 'です', 'score': 0.997816801071167, 'entity': 'O', 'index': 9},
  {'word': '。', 'score': 0.9979799389839172, 'entity': 'O', 'index': 10}]]
```
