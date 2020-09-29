---
title : "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks"
date : 2020-10-04
published : false
---

[논문 링크](https://arxiv.org/pdf/1907.12461v2.pdf), [Github 링크](https://github.com/huggingface/transformers)

# Abstract
대규모 신경망은 unsupervised 사전 훈련은 최근 자연어 처리에 큰 파장을 일으켰다. 공개적인 체크 포인트에서 시작하여 NLP 실무자들은 상당한 양의 컴퓨팅 시간을 절약하면서 여러 벤치 마크에서 최첨단 기술을 적용했다. 지금까지 주로 자연어 이해 작업에 중점을 두었다. 논문에서는 시퀀스 생성을 위한 사전 훈련된 체크 포인트의 효능을 보여준다. 우리는 공개적으로 사용 가능한 사전 훈 된 BERT, GPT-2, RoBERTa 체크 포인트와 호환되는 Transformer 기반 시퀀스-투-시퀀스 모델을 개발하고 인코더와 디코더를 모두 사용하여 모델을 초기화하는 유틸리티에 대한 광범위한 연구를 했다. 체크 포인트. 이 모델은 기계 번역, 텍스트 요약, 문장 분할, 문장 융합에 대한 새로운 최고 결과를 얻었다.
