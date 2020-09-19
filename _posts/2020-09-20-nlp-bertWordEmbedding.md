---
title : "BERT Word Embedding 튜토리얼 + 한국어처리"
date : 2020-09-20
categories : nlp
---
BERT word Embedding 튜토리얼을 소개한다. 이번 포스팅에서는 [원문](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)을 번역하고 한국어에 적용해본다.

# Contents
- Contents
- Introduction

# Introduction

### History
2018년, 전이학습(transfer learning)의 등장으로 NLP 분야는 크게 발전했다. Allen AI의 ELMO, OpenAI의 Open-GPT 및 Google의 BERT가 대표적인 모델이다. 
이 모델들은 최소한의 task별 fine-tuning으로 여러 NLP 과제에서 가장 좋은 성능을 냈다. 
NLP 커뮤니티에 쉽게(데이터와 컴퓨팅 시간을 줄임) 미세조정(fine-tuned) 맟 구현하여 생성할 수 있는 pre-trained 모델을 제공했다. 
하지만 전이학습에 대한 확실한 이해가 없어서 실무에 제대로 사용되지 않고 있다고 한다.

### What is BERT?
BERT(Bidirectional Encoder Representations from Transformers)는 2018년 말에 공개된 전이학습 모델이다. 사전 학습된 모델은 무료로 공개되어 있고, 
이 모델을 다운받아 텍스트 데이터에서 고품질 언어 기능을 추출하거나 특정 문제(분류, 엔티티 인식, 질문 답변, 등)를 해결하도록 미세조정 할 수 있다.

### Why BERT embeddings?
이 포스팅에서는 BERT를 사용하여 텍스트 데이터에서 특징, 즉 단어 및 문장 임베딩 벡터를 추출하는 방법을 설명하고자한다. 우선 단어와 문장 임베딩 벡터로 무엇을 할 수 있을까?
  - 임베딩은 키워드/검색어 확장, 의미 찾기 및 정보 검색에 유용하다. 예를 들어, 고객의 질문(검색)을 이미 답변된 질문이나 잘 문서화된 검색과 비교하려는 경우, 
  임베딩 벡터를 사용하면 키워드나 구문이 겹치지 않더라도 고객의 의도와 일치하는 결과를 찾을 수 있다.
  - 임베딩 벡터는 다운 스트림 모델에서 고품질 입력 피처로 사용된다. 
  LSTM 또는 CNN과 같은 NLP 모델에는 숫자 벡터 형식의 입력이 필요하며, 이는 일반적으로 어휘 및 품사와 같은 기능을 숫자로 변환하는 것을 의미한다. 
  이전에는 단어가 고유한 인덱스 값(원-핫 인코딩)으로 표현되거나 Word2Vec 또는 Fasttext와 같은 모델에서 생성된 고정 길이 임베딩과 어휘 단어가 일치하는 neural word embeddings으로 더 유용하게 표현되었다.

BERT는 Word2Vec과 같은 모델에 비해 문맥을 고려한 임베딩이 된다는 장점이 있다. 
Word2Vec은 단어가 나타나는 문맥에 관계없이 각 단어가 고정된 표현을 가지지만, BERT는 주변 단어에 의해 동적으로 변하는 단어 표현을 생성한다. 예를 들어 다음 두 문장이 주어진다면 :
  - 배를 타고 여행을 간다.
  - 추석에 먹은 배가 맛있었다.
Word2Vec은 두 문장의 "배"라는 단어에 대해 동일한 단어 임베딩을 생성하는 반면 BERT에서는 "배"에 대한 단어 임베딩이 문장마다 다르다.
다의어를 포착하는 것 외에도 상황에 맞는 단어 임베딩은 더 정확한 feature representation을 생성하는 다른 형태의 정보를 포착하여 더 나은 성능을 낸다. 


# 1. Loading Pre-Trained BERT
Hugging Face로 BERT 용 PyTorch 인터페이스를 설치합니다. 
(이 라이브러리에는 OpenAI의 GPT 및 GPT-2와 같은 사전 학습된 다른 언어 모델에 대한 인터페이스가 포함되어 있다.)

이 튜토리얼에서는 PyTorch를 사용한다. high-level API는 사용하기 쉽지만 작동 방식에 대한 통찰력을 제공하지 않고, tensorflow는 설정해야할 사항이 많다. 하지만 BERT를 사용하다보면 tensorflow를 사용할 일이 많다.

Google Colab에서 코드를 실행할 때, 다시 연결할 때만다 라이브러리를 설치해야한다.
```Python
!pip install transformers
```

이제 pytorch, pre-trained BERT, BERT tokenizer를 불러와야한다. 

BERT 모델은 Google의 사전 학습된 모델로 다양한 장르의 도서가 10,000 개 이상 포함된 데이터 세트 인 Wikipedia, Book Corpus에서 긴 시간동안 학습된 것이다. 이 모델은 NLP의 여러 과제에서 최고 점수를 달성했다(약간의 모델 수정 필요). Google이 공개한 여러 개의 BERT 중 원문에서는 ```bert-base-uncased```를 사용했지만, 이 포스팅에서는 한국어 처리를 위해 ```bert-base-multilingual-uncased```를 선택했다. 더 다양한 모델을 확인하고 싶다면, [여기](https://huggingface.co/transformers/pretrained_models.html)를 참고하자.

```transformers```는 BERT를 다른 작업(토큰 분류, 텍스트 분류 등)에 적용하기 위해 여러 클래스를 제공한다.
이번 포스팅에서는 단어 임베딩이 목적이기 때문에, 출력이 없는 기본 ```BertModel```을 사용한다. 

# 2. Input Formatting
BERT는 특정 형식의 입력 데이터를 필요로 한다.
1. **special token** ```[sep]``` : 문장의 끝을 표시하거나 두 문장의 분리
2. **special token** ```[CLS]``` : 
3. token : 

다행히도 ```transformers``` 인터페이스는 위의 모든 사항을 처리한다(tokenizer.encode_plus 함수 사용).
하지만 이 포스팅은 BERT 작업을 소개하기 위한 것이므로 (대부분)수동으로 이러한 단계를 진행한다.

```tokenizer.encode_plus```를 사용하는 예는 [여기](http://mccormickml.com/2019/07/22/BERT-fine-tuning/)에서 문장 분류에 대한 게시물을 참조하길 추천한다.

## 2.1. Special Tokens


## 2.2. Tokenization
BERT는 자체 토크나이저를 제공한다. 아래 문장을 어떻게 처리하는지 살펴보자.
```Python
text = "임베딩을 시도할 문장이다."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print(tokenized_text)
```
```
# ------ output ------- #
['[CLS]', '이', '##ᆷ', '##베', '##디', '##ᆼ을', '시', '##도', '##할', 'ᄆ', '##ᅮᆫ', '##장이', '##다', '.', '[SEP]']
```
```
'이', '##ᆷ', '##베', '##디', '##ᆼ을'
```
BERT 토크나이저가 WordPiece 모델로 생성되었기 때문에 위와 같은 결과가 나온다. 이 모델은 언어 데이터에 가장 적합한 개별 문자, 하위단어(subwords) 및 단어의 고정 크기 어휘를 탐욕스럽게(greedily) 만든다. BERT 토크나이저 모델의 어휘 제한 크기가 30,000 개이므로 WordPiece 모델은 모든 영어 문자와 모델이 훈련된 영어 말뭉치에서 발견되는 ~ 30,000 개의 가장 일반적인 단어 및 하위 단어를 포함하는 어휘를 생성했습니다. 이 어휘에는 다음 네 가지가 포함된다. :

  1. 전체 단어
  2. 단어의 앞에 또는 분리되어 발생하는 하위 단어 ( "embeddings"에서와 같이 "em"에는 "go get em"에서와 같이 독립형 문자 "em"시퀀스와 동일한 벡터가 할당 됨)
  3. 단어 앞에 있지 않은 하위 단어. 이 경우를 나타내기 위해 '##'이 앞에 붙는다.
  4. 개별 문자(individual character)

이 모델에서 단어를 토큰화하기위해 토크나이저는 먼저 전체 단어가 어휘에 있는지 확인한다. 그렇지 않은 경우 단어를 어휘에 포함 된 가능한 가장 큰 하위 단어로 나누고 마지막 수단으로 단어를 개별 문자로 분해한다. 이 때문에 우리는 항상 최소한 개별 문자의 모음으로 단어를 표현할 수 있다.

결과적으로 'OOV' 또는 'UNK'와 같은 포괄 토큰에 어휘 밖의 단어를 할당하는 대신 어휘에 포함되지 않은 단어는 임베딩을 생성할 수있는 하위 단어 및 문자 토큰으로 분해된다.

따라서 오버로드 된 알 수 없는 어휘 토큰에 "임베딩"및 기타 모든 어휘 단어를 할당하는 대신 하위 단어 토큰 [ 'em', '## bed', '## ding', '## s'로 분할합니다. ] 원본 단어의 문맥상 의미 중 일부를 유지한다. 이러한 하위 단어 임베딩 벡터를 평균하여 원래 단어에 대한 근사 벡터를 생성할 수도 있다. 
(WordPiece에 대한 자세한 내용은 [원본 논문](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)과 Google의 [Neural Machine Translation System](https://arxiv.org/pdf/1609.08144.pdf)을 참고)

다음은 어휘에 포함 된 토큰의 몇 가지 예이다. 두 개의 해시로 시작하는 토큰은 하위 단어 또는 개별 문자다.
```Python
list(tokenizer.vocab.keys())[20000:20020]
```
```
# ------ output ------- #
['weltkrieg',
 '##었다',
 'dock',
 'maakte',
 'бас',
 'mannschaft',
 '##ξη',
 '##list',
 'holy',
 '##nze',
 'dun',
 'sien',
 'hanet',
 'општина',
 '2015년',
 'dice',
 'motion',
 'ancienne',
 'hora',
 'lama']
```
multilingual 모델이기 때문에 다양한 언어가 포함되어 있다. 종종 한국어도 보인다. 텍스트를 토큰으로 분리한 후 문자열 목록의 문장을 어휘 목록으로 변환해야한다. 

```Python
# Define a new example sentence with multiple meanings of the word "bank"
text = "배를 타고 여행을 간다." \
       "추석에 먹은 배가 맛있었다."

# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
```
```
# ------ output ------- #
[CLS]           101
ᄇ             1,170
##ᅢ를       73,446
ᄐ             1,179
##ᅡ고        67,384
ᄋ             1,174
##ᅧ          46,069
##행을     91,480
가           20,966
##ᆫ다        32,407
.               119
ᄎ             1,177
##ᅮ          46,188
##석        40,482
##에         10,609
ᄆ             1,169
##ᅥ          33,645
##ᆨ은       34,653
ᄇ             1,170
##ᅢ          26,179
##가         11,376
ᄆ             1,169
##ᅡ          25,539
##ᆺ이        80,054
##ᆻ          97,104
##었다      20,001
.               119
[SEP]           102
```

## 2.3. Segment ID
BERT는 두 문장을 구별하기 위해 1과 0을 사용하여 문장 쌍을 학습하고 예상한다.
즉, 토큰화된 텍스트의 각 토큰에 대해 어떤 문장에 속하는지 지정해야한다 : 문장 0(0 리스트) 또는 문장 1(1 리스트).
우리의 목적을 위해 단일 문장 입력에는 1 리스트만 필요하므로 입력 문장의 각 토큰에 대해 1로 구성된 벡터를 생성한다.
```Python
# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

print (segments_ids)
```
```
# ------ output ------- #
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```


# 3. Extracting Embeddings
## 3.1. Running BERT on our text
데이터를 토치 텐서(torch tensor)로 변환하고 BERT 모델을 호출해야한다. BERT PyTorch 인터페이스에서는 데이터가 Python 리스트 아닌 토치 텐서가 필요하므로 이번 장에서 변환한다. - 이것은 모양이나 데이터를 변경하지 않는다.
```Python
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
```
```from_pretrained```를 호출하면 웹에서 모델을 다운로드한다. ```bert-base-multilingual-uncased```를 로드하면 로깅에 인쇄된 모델의 정의를 볼 수 있다. 이 모델은 12개의 레이어로 구성된 심층 신경망이다! 레이어와 그 기능에 대한 설명은이 게시물의 범위를 벗어나므로 건너뛴다.

```model.eval()```은 학습 모드가 아닌 평가 모드로 모델을 설정한다. 이 경우 평가 모드는 훈련에 사용되는 드롭아웃 정규화(dropout regularization)를 해제한다.
<details markdown="1">
<summary>bert-base-multilingual-uncased 모델 </summary>
<hr>
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(105879, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (2): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (3): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (4): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (5): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (6): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (7): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (8): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (9): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (10): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (11): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
<hr>
</details>

