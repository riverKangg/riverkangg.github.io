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
