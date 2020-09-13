---
title : "[Python/NLP] 용어 사전 만들기"
date : 2020.09.10
categories : NLP
tags:
- NLP
- dictionary
puublished : false
---

- 목적 : 특수한 분야의 용어를 잘 인식할 수 있도록 용어 사전을 구축하기
- 구축과정
  1. sentencePiece
  2. 

### BERT - Multilingual
[BERT multilingual Github](https://github.com/google-research/bert/blob/master/multilingual.md)
BERT는 여러 언어에 적용이 가능한 모델이지만, 한국어/중국어/일본어에 대해서는 어려운 점이 있다.

#### BERT의 한국어 처리 문제점
BERT는 토큰화를 위해 110k shared WordPiece 어휘를 사용한다. 단어 수는 데이터와 같은 방식으로 가중치가 부여되므로 리소스가 적은 언어는 일부 요인에 의해 가중치가 높아질 수 있다. 의도적으로 입력 언어를 나타내는 마커를 사용하지 않는다.(제로 샷 훈련이 작동 할 수 있도록).
중국어(일본어 간지, 한국어 한자)에는 공백 문자가 없기 때문에 WordPiece를 적용하기 전에 CJK 유니 코드 범위의 모든 문자 주위에 공백을 추가합니다. 이는 중국어가 효과적으로 문자 토큰화됨을 의미한다. CJK 유니 코드 블록에는 중국어 원본 문자만 포함되며 다른 모든 언어와 마찬가지로 공백 + WordPiece로 토큰화 된 한글 또는 가타카나/히라가나 일본어는 포함되지 않는다.
다른 모든 언어의 경우 (a) 소문자 + 악센트 제거, (b) 구두점 분할, (c) 공백 토큰 화와 같이 영어와 동일한 레시피를 적용합니다. 악센트 표시는 일부 언어에서 상당히 중요한 것을 이해하지만, 어휘 감소의 이점 이를 보완한다. 일반적으로 BERT의 강력한 컨텍스트 모델은 강조 표시를 제거하여 발생하는 모호성을 보완해야한다.

- Transfer Learning : 학습 데이터가 부족한 분야의 모델 구축을 위해 데이터가 풍부한 분야에서 훈련된 모델을 재사용하는 머신러닝 학습 기법
- zero-shot learning : Transfer learning에서 발전된 기계학습 기법. 라벨링이 필요없음. 훈련된 데이터와 훈련되지 않은 데이터의 특징을 잡아냄.

### koBert

[KoBERT Github](https://github.com/SKTBrain/KoBERT#why)

SKTBrain에서는 이러한 문제를 해결한 BERT를 공개했다.

```
