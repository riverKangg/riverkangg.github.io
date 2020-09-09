---
title : "[논문리뷰] SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
date : 2020-09-09
categories : 논문리뷰
tags :
- 추천알고리즘
- 논문리뷰
use_math: true
---

수정중

[SentencePiece Paper](https://arxiv.org/pdf/1808.06226.pdf/)
[SentencePiece Github](https://github.com/google/sentencepiece/)



# Abstract

SentencePiece는 언어에 구애받지 않는 subword tokenizer와 detokenizer이다. 신경망기계번역을 포함한 신경망 기반의 텍스트 처리로 구성되어 있다.
센텐스피스는 C++과 Python을 제공한다. 이전의 서브워드 분할 인풋은 워드시퀀스로 pre-tokenized 되어야 한다. 하지만 센텐스피스는 기존 문장을 바로 서브워드 모델로 학습시킬 수 있다. 따라서 완전한 **end-to-end**와 언어에 구애받지 않는 시스템을 만들 수 있다. 

NMT 모델을 영어-일본어 기계번역으로 검증 실험한다. 



# Introduction

NMT가 end-to-end 번역을 하더라도 아직 특정 언어에 한정되어 있고, pre-와 postprocessor에 의존한다. 전통적인 통계기계번역(SMT, Statistical machine translation) 시스템은 유럽언어들에 맞춰 만들어져있다. 이는 한국어, 중국어, 일본어 같은 비분할(non-segmented) 언어들에는 단어 분할기를 각각 만들어야 한다. 

- 간단한
- 효율적
- 재사용 가능한

1. Byte-pair-encoding(BPE)
2. uni-gram language model



# 2. System Overview

센텐스피스는 네가지 요소로 구성된다.

- **Nomarlizer** : a 
- **Trainer**
- **Encoder**
- **Decoder** : 서브워드 시퀀스를 normalized된 텍스트로 바꿈




# 3. Library Design

## 3.1 Lossless Tokenization

Decoder를 Encoder의 역연산으로 이용한다.
<center>
   Decode(Encoder(Normalize( *text* ))) = Normalize( *text* )
</center>  

정규화 된 텍스트를 재현하기 위한 모든 정보는 인코더의 출력에 보존됩니다. 무손실 토큰화는 유니코드 문자의 시퀀스를 입력 텍스트로 다룬다. 

1. 공백 -> **\_**(U+2581)
      - detokenize 하는 파이썬 코드이다.
      ```
      # detokenize code
      detok = ''.join(tokens).replace('_', ' ')
      ```

2. **\@\@** : 단어 경계 간 마커



## 3.2 Efficient subword training and segmentation

- pre-tokenization : 




## 3.3 Vocabulary id management




## 3.4 Customizable character normalization

문자 정규화(character normalization)은 
