---
title : "[논문리뷰] Abstractive Summarization of Spoken and Written Instructions with BERT"
date : 2020-09-27
categories : 논문리뷰
tags :
- 자연어처리
- 논문리뷰
published : false
---

# ABSTRACT

# METHODOLOGY
## 3.1 Data Collection

- **CNN/DailyMail Dataset**
- **Wikihow Dataset**
- **How2 Dataset**

## 3.2 Preprocessing
input 데이터가 다양하고 복잡하기 때문에, 일반적인 포맷으로 데이터를 정렬하는 전처리 파이프라인을 만들었다. 모델 학습에 영향을 준 구두점, 잘못된 단어, 관계없는 소개에 대한 문제가 있었다. 이런 문제로, 모델이 text segment를 잘못 나누고 좋지 않은 요약 결과를 냈다. 예외적인 경우, 모델은 요약된 결과를 내지 못했다. 사람이 작성한 요약의 유창성과 일관성을 유지하기 위해, 다음과 같이 문장 구조를 정제했다. 오픈소스 라이브러리(spacy, nltk)로 entitiy detection을 실행한다. nltk는 introduction을 제거하고, 요약모델의 인풋을 익명화한다. 문장을 나누고 모든 데이터셋에 Stanford Core NLP toolkit을 토큰화하고, [See et.al](https://arxiv.org/pdf/1704.04368.pdf)과 같은 방법으로 데이터를 전처리한다. 

## 3.3 Summerization models
[Text Summarization with Pretrained Encoders](https://www.aclweb.org/anthology/D19-1387.pdf)에 제안된 BertSum 모델을 사용한다. 추출(Extractive) 요약과 추상(Abstractive) 요약을 동시에 포함하고 있고, 이는 BERT를 기반으로 문서 수준의 인코더를 사용한다. transformer 구조는 무작위로 초기화된 Transformer decoder로 pretrained BERT 인코더에 적용된다. 이는 두가지 학습률(learning rate)를 사용한다.: 인코더에는 작은 학습률을 사용하고, 디코더에는 학습을 향상시키기 위해 더 큰 학습률을 사용한다.

How2 데이터 셋에서 5,000개 영상 샘플에 추출모델을 학습시켜 베이스라인을 만든다(4-GPU Linux 사용). 처음에는 BERT-base-uncased를 10,000 스텝에 적용하고, 가장 좋은 성능을 낼 수 있는 epoch 크기를 골라서 요약 모델과 BERT 레이어 미세조정(fine tuned)한다. 이 초기 모델에 How2와 WikiHow에 개별적으로 추상 요약 모델을 추가적으로 학습한다.

추상 요약 모델의 가장 좋은 버전은 CNN/DailyMail, Wikihow, How2을 모두 합친 데이터 셋으로 만들어졌다. 이 데이터 셋은 535,527개의 예시와 210,000개의 스탭으로 되어있다. 사이즈 50의 배치 크기를 사용하고, 20 에폭으로 모델을 학습한다. 데이터 셋의 순서를 조정하여, 요약의 유창성을 향상시킬 수 있다. 이전의 연구에서 언급한 것처럼, 기존의 모델은 1억 8천만개 이상의 모수가 있고, 
