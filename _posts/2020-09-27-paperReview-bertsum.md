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
요약은 흐름의 자발성, 유창하지 않음과 일반적으로 서면 텍스트에서 발생하지 않는 기타 문제로 인해 어려운 문제다. 우리의 연구는 BERTSum 모델을 대화 언어에 처음 적용한 것이다.
정원 가꾸기 및 요리에서 소프트웨어 구성 및 스포츠에 이르기까지 다양한 주제에 대해 설명 된 교육용 비디오의 추상 요약을 생성한다.
어휘를 풍부하게하기 위해 전이 학습을 사용하고 서면 및 구어체 영어로 된 몇 가지 대규모 교차 도메인 데이터 세트에서 모델을 사전 학습한다.

#### CSS CONCEPTS
- **Human-centered computing** &#8594; *Accessibility system and tools;*
- **Computing methodologies** &#8594; **Information extraction*

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

## 4.1 Training
BertSum 모델은 CNN/DailyMail 데이터 셋에서 가장 좋은 성능을 냈다. BertSum 모델은 추출 요약과 추상 요약 모두를 지원한다.  베이스라인은 CNN/DailyMail에서 How2 video로 사전 학습된 추출 요약 BertSum 모델에서 얻은 것이다. 그러나 모델은 매우 낮은 점수를 냈다. 모델에서 생성한 요약은 일관되지 않고 반복적이며 정보가 없었다. 안좋은 성능에도 불구하고, 모델은 How2 video 내의 건강 하위 도메인에서 더 잘 수행되었다. CNN/DailyMail에 의해 생성된 뉴스에서 과도한 보도의 증상이라고 설명했다. 추출 요약은 가장 좋은 모델이 아니다.: 대부분의 YouTube 영상은 캐주얼한 대화 스타일인 반면 요약은 더 형식적이다. 성능 향상을 위해 추상 요약으로 바꿨다.

추출 요약 모델은 사전 학습된 BERT 인코더와 무작위로 초기화 된 Transformer 디코더를 합친 인코더-디코더 구조를 사용한다. 인코더 부분이 매우 낮은 학습률로 거의 동일하게 유지되는 특수 기술을 사용하고 디코더가 더 잘 학습할 수 있도록 별도의 학습률을 생성한다. 일반화 가능한 추상 모델을 만들기 위해 먼저 대규모 뉴스 코퍼스를 학습하여, 구조화 된 텍스트를 이해하도록 했다. 그런 다음 모델을 How-To 도메인에 노출하는 Wikihow를 도입했다. 마지막으로 How2 데이터 세트에 대해 학습하고 검증하여 모델의 초점을 선택적으로 구조화된 형식으로 좁혔다. 순서가 지정된 훈련 외에도 무작위로 균질한 샘플 세트를 사용하여 모델 훈련을 실험했다. 순서가 지정된 샘플로 훈련하는 것이 무작위 샘플보다 더 좋은 결과를 냈다.

<p align="center">
  <img src="https://raw.githubusercontent.com/riverKangg/riverkangg.github.io/master/_posts/image/2020-09-27-fig4.png" width=400, alt='fig4:Cross Entropy:Training vs Validation'>
</p>
위 교차 엔트로피 차트를 보면, 과적합이나 과소적합되지 않았음을 확인할 수 있다. 훈련 및 검증 라인의 수렴으로 좋은 적합성을 나타낸다.
<p align="center">
  <img src="https://raw.githubusercontent.com/riverKangg/riverkangg.github.io/master/_posts/image/2020-09-27-fig4.png" width=400, alt='fig5:Accuracy:Training vs Validation'>
</p>
위 그림은 훈련 및 검증 세트에 대한 모델의 정확도 메트릭을 보여준다. 모델은 훈련 데이터에 대해 How2 데이터로 검증한다. 모델은 더 많은 단계를 통해 예상대로 향상한다.

## 4.2 Evalutation
CNN/DailyMail로 학습된 BertSum은 해당 데이터셋에 적용될 때 최상의 결과를 냈지만, How2 데이터셋에 테스트 했을 때는 성능이 좋지 않고 일반화가 되지 않았다(표3-1행). 데이터를 보면 첫번째나 두번째 문장을 선택하는 경향이 있다. 텍스트에서 introductions를 지우는게 ROUGE 스코어를 증가하는데 도움이 됐다. 3.2에서 설명한 전처리를 적용한 후, 몇 가지 ROUGE 포인트를 개선했다. 또 모델에 익숙하지 않은 희귀 단어에서 발생하는 것을 관찰해서 output에 단어 중복 제거를 추가해서 모델을 개선했다. 22.5 ROUGE-1 F1과 20 ROUGE-L F1 보다 더 높은 점수를 받지 못했다(초기 점수는 CNN/DailyMail로 학습하고 How2로 테스트 함). 개별 요약의 점수와 텍스트를 봤을 때, 의학과 같은 일부 주제에서 더 나은 성과를 내지만 스포츠와 같은 다른 주제에서는 낮은 점수를 받았다.

사전 학습된 비디오 스크립트와 뉴스 스토리의 대화 스타일 차이가 모델 output에 영향을 준다. CNN/DailyMail에 대해 사전 훈련된 추출 요약 모델의 초기 적용에서 스타일 오류가 뚜렷하게 나타났다. 요약을 생성할 때, 초기 소개 문장이 중요하다고 간주했다([15]에서 N-lead로 표현됐고, 여기서 N은 중요한 첫 문장의 수입니다). 모델은 "hi"와 "hello, the is <first and last name>"과 같이 짧고 간결한 단어로 요약을 생성했다.
  
How2에 추상 BertSum을 재학습하는 것은 흥미롭고 기대하지 않았던 결과를 보여줬다.-
