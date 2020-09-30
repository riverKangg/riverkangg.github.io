---
title : "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks"
date : 2020-10-04
published : false
---

[논문 링크](https://arxiv.org/pdf/1907.12461v2.pdf), [Github 링크](https://github.com/huggingface/transformers)

# Abstract
대규모 신경망은 unsupervised 사전 훈련은 최근 자연어 처리에 큰 파장을 일으켰다. 공개적인 체크 포인트에서 시작하여 NLP 실무자들은 상당한 양의 컴퓨팅 시간을 절약하면서 여러 벤치 마크에서 최첨단 기술을 적용했다. 지금까지 주로 자연어 이해 작업에 중점을 두었다. 논문에서는 시퀀스 생성을 위한 사전 훈련된 체크 포인트의 효능을 보여준다. 우리는 공개적으로 사용 가능한 사전 훈 된 BERT, GPT-2, RoBERTa 체크 포인트와 호환되는 Transformer 기반 시퀀스-투-시퀀스 모델을 개발하고 인코더와 디코더를 모두 사용하여 모델을 초기화하는 유틸리티에 대한 광범위한 연구를 했다. 체크 포인트. 이 모델은 기계 번역, 텍스트 요약, 문장 분할, 문장 융합에 대한 새로운 최고 결과를 얻었다.

# 4 Experiments and Results
## 4.1 Sentence Fusion
Sentence Fusion은 여러 문장을 하나의 일관된 문장으로 결합하는 문제다. 450만 개의 DiscoFuse 데이터 셋(Geva et al., 2019)의 "balanced Wikipedia"를 트레인 셋으로 사용한다. 평가 셋에는 50k이다. 평가 셋의 크기가 작기 때문에 작은 변화도 통계적으로 유의미하다. 이러한 이유로 우리는 논문 끝에 설명된 추가 실험을 위해 이 데이터 세트를 단독으로 선택했다. 훈련은 전역 배치 크기 256으로 30만 단계에 대해 수행되었다. 입력과 출력은 훈련, 평가 및 테스트 데이터의 100%를 포함하는 128의 길이로 채워진다. SARI(Xu et al., 2016) 및 정확한 일치 정확도를 보인다. 결과는 다음 표에서 볼 수 있다. Geva et al.(2019)의 이전 최신 결과는 Vaswani et al.(2017)의 바닐라 트랜스포머 모델을 7개의 레이어로만 사용했다. 초기화 된 인코더가 있는 모든 모델은 86.9 (BERT2RND vs. RND2RND)에 비해 SARI 점수가 89.3 점으로 기준선을 크게 능가했다. 더 작은 훈련 세트에 대한 영향을 측정하기 위해 훈련 데이터를 무작위로 10% 및 1%, 즉 각각 450k 및 45k 훈련 예제로 하위 샘플링한다. 첫째, 훈련 데이터의 10% (RND2RND vs ROBERTASHARE)로만 훈련하더라도 기준과 비슷한 성능이 달성되었음을 알 수 있다. 둘째, 무작위로 초기화 된 매개 변수 (BERT2BERT vs BERT2RND)가 적은 훈련 데이터 설정의 1% 만 사용할 때 더 나은 성능을 발휘한다. 최고 성능의 12 계층 설정은 SARI 점수가 89.9 인 ROBERTA2GPT이며, SARI 점수가 90.3 인 ROBERTASHARE의 24 계층 설정보다 성능이 뛰어나다.

## 4.2 Split and Rephrase
