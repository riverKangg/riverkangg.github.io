---
title : "BERT Word Embedding 튜토리얼 + 한국어처리"
date : 2020-09-20
categories : nlp
---
BERT word Embedding 튜토리얼을 소개한다. 이번 포스팅에서는 [원문](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)을 번역하고 한국어에 적용해본다.

# Contents
- Contents
- Introduction
  - History
  - What is BERT?
  - Why BERT embeddings?
- 1. Loading Pre-Trained BERT
- 2. Input Formatting
  - 2.1. Special Tokens
  - 2.2. Tokenization
  - 2.3. Segment ID
- 3. Extracting Embeddings
  - 3.1. Running BERT on our text
  - 3.2. Understanding the Output
  - 3.3. Creating word and sentence vectors from hidden states
    - Word Vectors
    - Sentence Vectors
  - 3.4. Confirming contextually dependent vectors
  - 3.5. Pooling Strategy & Layer Choice
- 4. Appendix
  - 4.1. Special tokens
  - 4.2. Out of vocabulary words
  - 4.3. Similarity metrics
  - 4.4. Implementations
    - Cite

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
BERT는 하나 또는 두 개의 문장을 입력으로 사용할 수 있으며 특수 토큰 ```[SEP]```을 사용하여 구분한다. ```[CLS]``` 토큰은 항상 텍스트 시작 부분에 나타나며 분류 작업에만 해당된다.

그러나 두 개의 토큰은 항상 필요하다. 그러나 우리가 문장이 하나 뿐이고 분류에 BERT를 사용하지 않더라도 마찬가지다. 이것이 BERT가 사전 훈련된 방법이며 BERT가 기대하는 것이다.

**2 Sentence Input:**
```
[CLS] 드디어 내일이 주말이다. [SEP] 날씨가 맑으면 공원에 가야겠다.
```
**1 Sentence Input:**
```
[CLS] 드디어 내일이 주말이다. [SEP]
```

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

BERT 토크나이저가 WordPiece 모델로 생성되었기 때문에 위와 같은 결과가 나온다. 이 모델은 언어 데이터에 가장 적합한 개별 문자, 하위단어(subwords) 및 단어의 고정 크기 어휘를 탐욕스럽게(greedily) 만든다. BERT 토크나이저 모델의 어휘 제한 크기가 30,000 개이므로 WordPiece 모델은 모든 영어 문자와 모델이 훈련된 영어 말뭉치에서 발견되는 ~ 30,000 개의 가장 일반적인 단어 및 하위 단어를 포함하는 어휘를 생성했습니다. 

이 어휘에는 다음 네 가지가 포함된다. :

  1. 전체 단어
  2. 단어의 앞에 또는 분리되어 발생하는 하위 단어 ( "embeddings"에서와 같이 "em"에는 "go get em"에서와 같이 독립형 문자 "em"시퀀스와 동일한 벡터가 할당 됨)
  3. 단어 앞에 있지 않은 하위 단어. 이 경우를 나타내기 위해 '##'이 앞에 붙는다.
  4. 개별 문자(individual character)

이 모델에서 단어를 토큰화하기위해 토크나이저는 먼저 전체 단어가 어휘에 있는지 확인한다. 그렇지 않은 경우 단어를 어휘에 포함 된 가능한 가장 큰 하위 단어로 나누고 마지막 수단으로 단어를 개별 문자로 분해한다. 이 때문에 우리는 항상 최소한 개별 문자의 모음으로 단어를 표현할 수 있다.

결과적으로 'OOV' 또는 'UNK'와 같은 포괄 토큰에 어휘 밖의 단어를 할당하는 대신 어휘에 포함되지 않은 단어는 임베딩을 생성할 수있는 하위 단어 및 문자 토큰으로 분해된다.

따라서 오버로드 된 알 수 없는 어휘 토큰에 "임베딩"및 기타 모든 어휘 단어를 할당하는 대신 하위 단어 토큰 \['이', '##ᆷ', '##베', '##디', '##ᆼ을'\]로 분할합니다. 원본 단어의 문맥상 의미 중 일부를 유지한다. 이러한 하위 단어 임베딩 벡터를 평균하여 원래 단어에 대한 근사 벡터를 생성할 수도 있다. 
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

*Note : 포스팅이 너무 길어져서 output을 삭제했다. 자세한 결과는 여기 [Colab notebook]()에 있다.*

다음으로 예제 텍스트에서 BERT를 평가하고 네트워크의 숨겨진 상태를 가져온다!

```torch.no_grad```는 PyTorch가 순방향 패스(forward pass)동안 컴퓨팅 그래프를 구성하지 않도록 한다.(여기서는 backprop를 실행하지 않기 때문에).-이는 메모리 소비를 줄이고 작업 속도를 약간 높일뿐이다.
```Python
# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers. 
with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]
```


## 3.2. Understanding the Output
```hidden_states``` 개체에 저장된 이 모델의 전체 은닉층은 약간 복잡하다. 이 개체에는 다음 순서로 4개의 차원이 있다.

  1. 레이어 번호 (13 레이어)
  2. 배치 번호 (1 문장)
  3. 단어 / 토큰 번호 (문장에서 22 개의 토큰)
  4. 숨겨진 유닛 / 기능 번호 (768 개 기능)
  
잠깐 13 레이어? BERT에는 12 개만 있지 않나? 첫 번째 요소는 입력 임베딩이고 나머지는 BERT의 12개 레이어 각각의 출력이므로 13이다.

```Python
print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
```
```
# ------ output ------- #
Number of layers: 13   (initial embeddings + 12 BERT layers)
Number of batches: 1
Number of tokens: 28
Number of hidden units: 768
```

주어진 레이어와 토큰에 대한 값의 범위를 간단히 살펴보자. 범위가 모든 레이어와 토큰에 대해 상당히 유사하다는 것을 알 수 있다.-대부분의 값이 [-2.5, 2.5] 사이에 있다.
```Python
# For the 5th token in our sentence, select its feature values from layer 5.
token_i = 5
layer_i = 5
vec = hidden_states[layer_i][batch_i][token_i]

# Plot the values as a histogram to show their distribution.
plt.figure(figsize=(10,10))
plt.hist(vec, bins=200)
plt.show()
```

계층별로 값을 그룹화하는 것은 모델에 적합하지만 단어 임베딩을 위해 토큰별로 그룹화하는 것이 좋다.

현재 차원 :
```
[# layers, # batches, # tokens, # features]
```
원하는 차원 :
```
[# tokens, # layers, # features]
```

다행히 PyTorch에는 텐서 차원을 쉽게 재배열 할 수 있는 ```permute```함수가 포함되어있다.

그러나 첫 번째 차원은 현재 Python list이다!
```Python
# `hidden_states` is a Python list.
print('      Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[0].size())
```
```
# ------ output ------- #
      Type of hidden_states:  <class 'tuple'>
Tensor shape for each layer:  torch.Size([1, 28, 768])
```

레이어를 결합해서 하나의 큰 텐서를 만든다.
```Python
# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
token_embeddings = torch.stack(hidden_states, dim=0)

token_embeddings.size()
```
```
torch.Size([13, 1, 28, 768])
```

"batches" 차원은 필요하지 않으므로 제거한다.
```Python
# Remove dimension 1, the "batches".
token_embeddings = torch.squeeze(token_embeddings, dim=1)

token_embeddings.size()
```
```
torch.Size([13, 28, 768])
```

마지막으로 ```permute```를 사용하여 "layers" 및 "tokens" 차원을 전환할 수 있다.
```Python
# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)

token_embeddings.size()
```
```
torch.Size([28, 13, 768])
```

## 3.3. Creating word and sentence vectors from hidden states
은닉층으로 무엇을 할 수 있을지 알아보자. 각 토큰에 대한 개별 벡터 또는 전체 문장의 단일 벡터 표현을 얻고 싶지만, 입력의 각 토큰에 대해 각각 768 크기의 13개의 개별 벡터가 있다.

개별 벡터를 얻으려면 일부 레이어 벡터를 결합해야한다. 그러나 어떤 레이어 또는 레이어 조합이 최상의 표현을 제공할까요?

안타깝게도 쉬운 답은 없다. 하지만 몇 가지 합리적인 접근 방식을 시도해볼 수 있다. 그 후이 질문에 대해 자세히 살펴볼 수 있는 몇 가지 유용한 리소스가 있다.

### Word Vectors
몇 가지 예를 들어 두 가지 방법으로 단어 벡터를 만들 수 있다.

먼저 마지막 4개의 레이어를 연결하여 토큰 당 단일 단어 벡터를 제공한다. 각 벡터의 길이는 ```4 x 768 = 3,072```입니다.
```Python
# Stores the token vectors, with shape [22 x 3,072]
token_vecs_cat = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:
    
    # `token` is a [12 x 768] tensor

    # Concatenate the vectors (that is, append them together) from the last 
    # four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    
    # Use `cat_vec` to represent `token`.
    token_vecs_cat.append(cat_vec)

print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
```
```
Shape is: 28 x 3072
```
다른 방법으로 마지막 4개의 레이어를 합산하여 단어 벡터를 만든다.
```Python
# Stores the token vectors, with shape [22 x 768]
token_vecs_sum = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:

    # `token` is a [12 x 768] tensor

    # Sum the vectors from the last four layers.
    sum_vec = torch.sum(token[-4:], dim=0)
    
    # Use `sum_vec` to represent `token`.
    token_vecs_sum.append(sum_vec)

print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
```
```
Shape is: 28 x 768
```

### Sentence Vectors
전체 문장에 대한 단일 벡터를 얻기 위해 여러 application-dependent 전략이 있지만, 간단한 접근 방식은 단일 768 크기의 벡터를 생성하는 각 토큰의 두 번째에서 마지막 숨겨진 레이어를 평균내는 것이다.
```Python
# `hidden_states` has shape [13 x 1 x 22 x 768]

# `token_vecs` is a tensor with shape [22 x 768]
token_vecs = hidden_states[-2][0]

# Calculate the average of all 22 token vectors.
sentence_embedding = torch.mean(token_vecs, dim=0)
```
```
Our final sentence embedding vector of shape: torch.Size([768])
```

## 3.4. Confirming contextually dependent vectors
```Python
for i, token_str in enumerate(tokenized_text):
  print (i, token_str)
```
```
# ------ output ------- #
0 [CLS]
1 ᄇ
2 ##ᅢ를
3 ᄐ
4 ##ᅡ고
5 ᄋ
6 ##ᅧ
7 ##행을
8 가
9 ##ᆫ다
10 .
11 ᄎ
12 ##ᅮ
13 ##석
14 ##에
15 ᄆ
16 ##ᅥ
17 ##ᆨ은
18 ᄇ
19 ##ᅢ
20 ##가
21 ᄆ
22 ##ᅡ
23 ##ᆺ이
24 ##ᆻ
25 ##었다
26 .
27 [SEP]
```

```Python
print('First 5 vector values for each instance of "배".')
print('')
print("배를 타다   ", str(token_vecs_sum[6][:5]))
print("배를 먹다  ", str(token_vecs_sum[10][:5]))
print("바다에 있는 배   ", str(token_vecs_sum[19][:5]))
```
```
First 5 vector values for each instance of "배".

배를 타다    tensor([-0.1956,  0.6169, -1.2606,  3.6393,  0.2309])
배를 먹다   tensor([-0.3030,  1.2585, -1.8328, -0.2811,  3.1500])
바다에 있는 배    tensor([-0.2556, -1.8705,  1.2312,  0.0592, -1.1682])
```
값이 다른 것을 볼 수 있지만 더 정확한 비교를 위해 벡터 간의 코사인 유사성을 계산한다.


## 3.5. Pooling Strategy & Layer Choice
단어 임베딩을 위한 추가 리소스이다.

**BERT Authors**
BERT 작성자는 명명 된 개체 인식 작업에 사용되는 BiLSTM에 입력 기능으로 다양한 벡터 조합을 제공하고 결과 F1 점수를 관찰하여 단어 삽입 전략을 테스트한다. 
[링크](http://jalammar.github.io/illustrated-bert/)

마지막 4개 레이어를 연결하면이 특정 작업에서 최상의 결과를 얻을 수 있었지만 다른 많은 방법이 가까운 순간에 나오며 일반적으로 특정 애플리케이션에 대해 다른 버전을 테스트하는 것이 좋다. 결과는 다를 수 있다.

이것은 BERT의 서로 다른 계층이 매우 다른 종류의 정보를 인코딩한다는 점을 지적함으로써 부분적으로 입증된다. 따라서 서로 다른 계층이 서로 다른 종류의 정보를 인코딩하기 때문에 애플리케이션에 따라 적절한 풀링 전략이 변경된다.

**Han Xiao’s BERT-as-service**
Han Xiao는 BERT를 사용하여 텍스트에 대한 단어 임베딩을 생성하기 위해 GitHub에서 [bert-as-service](https://github.com/hanxiao/bert-as-service)라는 오픈 소스 프로젝트를 만들었다. Han은 이러한 임베딩을 결합하는 다양한 접근 방식을 실험하고 프로젝트의 [FAQ](https://github.com/hanxiao/bert-as-service#speech_balloon-faq)에서 몇 가지 결론과 근거를 공유한다.

```bert-as-service```는 기본적으로 모델의 마지막에서 두 번째 계층의 출력을 사용한다.

요약하면 다음과 같다. :


# 4. Appendix
## 4.1. Special tokens
```[CLS]```가 분류 작업에 대한 "aggregate representation" 역할을 하지만, 고품질 문장 임베딩 벡터를 위한 최선의 선택은 아니다. BERT 작성자 Jacob Devlin에 따르면: BERT는 의미있는 문장 벡터를 생성하지 않기 때문에, 벡터가 무엇을 의미하는지 확실하지 않다. 이것은 문장 벡터를 얻기 위해 단어 토큰에 대한 평균 풀링을 수행하는 것처럼 보이지만 이것이 의미있는 문장 표현을 생성한다는 말은 없었다.

(그러나 [CLS] 토큰은 모델이 미세 조정 된 경우 의미가 있습니다. 여기서이 토큰의 마지막 은닉층은 시퀀스 분류를 위한 "문장 벡터"로 사용된다.)

## 4.2. Out of vocabulary words
여러 문장과 문자 수준 임베딩으로 구성된 oov(out of vocabulary) 어휘 중, 임베딩을 복구하는 방법을 찾는 문제가 있다. 임베딩 평균화는 가장 간단한 솔루션 (빠른 텍스트와 같은 하위 단어 어휘를 사용하는 유사한 임베딩 모델에 의존하는 솔루션)이지만, 하위 단어 임베딩의 합계와 단순히 마지막 토큰 임베딩 (벡터는 상황에 따라 다름)을 취하는 것도 방법이다.

## 4.3. Similarity metrics
이 임베딩은 문맥에 따라 달라지기 때문에, 단어 수준 유사성 비교가 BERT 임베딩에 적합하지 않다.

사용된 유사성 메트릭(similarity metric)에 따라, 

## 4.4. Implementations
이 노트북의 코드를 자체 애플리케이션의 기초로 사용하여 텍스트에서 BERT 기능을 추출할 수 있다.
그러나 공식 [tensorflow](https://github.com/google-research/bert/blob/master/extract_features.py)와 잘 알려진 [pytorch]()가 이미 존재한다.
또한 [bert-as-a-service](https://github.com/hanxiao/bert-as-service)는이 작업을 고성능으로 실행하도록 특별히 설계된 우수한 도구다.
작성자는 도구구현에 신경썼으며, 리소스 관리 및 풀링 전략과 같이, 사용자가 겪는 미묘한 문제를 해결하는데 도움이 되는 문서(일부는 이 가이드를 만드는데 사용됨)를 제공한다.
### Cite
Chris McCormick and Nick Ryan. (2019, May 14). BERT Word Embeddings Tutorial. Retrieved from <http://www.mccormickml.com>
