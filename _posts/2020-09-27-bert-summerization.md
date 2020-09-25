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

추상 요약 모델의 가장 좋은 버전은 CNN/DailyMail, Wikihow, How2을 모두 합친 데이터 셋으로 만들어졌다. 이 데이터 셋은 535,527개의 예시와 210,000개의 스탭으로 되어있다. 사이즈 50의 배치 크기를 사용하고, 20 에폭으로 모델을 학습한다. 데이터 셋의 순서를 조정하여, 요약의 유창성을 향상시킬 수 있다. 이전의 연구에서 언급한 것처럼, 기존의 모델은 1억 8천만개 이상의 모수가 있고,  인코더와 디코더 각각 $\beta_1=0.9$, $\beta_2=0.999$로 Adam 옵티마이저를 사용한다. learning rate는 인코더는 0.002이고, 디코더는 0.2를 사용한다. 이것은 디코더가 안정되는 동안, 인코더가 더 정확한 기울기로 훈련되었는지 확인한다. 결과는 Section 4에 나와있다.

사람이 배우는 것과 같이, 모델에서도 훈련 순서가 중요하다고 가정한다. 자연어처리에 [curriculum learning](https://dblp.uni-trier.de/db/conf/icml/icml2009.html#BengioLCW09)을 적용하는 방법은 관심이 증가했다. 더 복잡하지만 예측 가능한 언어 구조4로 이동하기 전, 고도로 구조화된 샘플을 학습한다. 텍스트 스크립트를 훈련한 후, 비디오 스크립트를 진행한다다. 이 스크립트는 임시 흐름과 대화 언어에 대한 추가적인 문제가 된다.

## 3.4 Scoring of results
결과는 추상 요약에서 일반적으로 사용하는 ROUGE를 이용해서 스코어링 된다. 요약이 잘되면 높은 ROUGE 점수가 나올거라 기대했지만, 결과에서는 요약이 잘 안됐지만 ROUGE가 높고, 요약이 잘됐지만 ROUGE 점수는 낮았다.(Figure 10)

Content F1 스코어링을 추가했다. 이 메트릭은 컨텐츠의 관련성에 초점을 맞췄다. ROUGE와 비슷하게, Content F1 스코어는 가중 f-score와 잘못된 단어 순서로 요약한다. 

작성된 요약없이 구절을 스코어링하려면, Python, Google Forms, Excel 스프레드 시트를 사용하여 평가 프레임워크로 사람의 판단을 조사했다. 사람의 판단이 포함된 요약들은 편향되지 않기 위해 랜덤하게 샘플링 된다. 사람과 기계가 생성한 요약 간에 비대칭 정보를 줄이기 위해 대문자를 제거했다. 두가지 질문을 했다.: AI와 인간이 생성한 설명과 구별하기 위한 튜링 테스트 질문이다. 두번째는 요약에 대한 품질을 선택하는 것이다. 다음은 명확성을 판단하기위한 기준의 정의이다.

  - 유창성 : 
  
요약 등급은 다음과 같다.: 1:나쁨 2:평균이하 3:평균 4:좋음 5:매우좋음

# 4. EXPERIMENTS AND RESULTS
