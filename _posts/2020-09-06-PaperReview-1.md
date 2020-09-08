---
title : "[논문리뷰] Deep Neural Networks for YouTube Recommendations"
date : 2020-09-06
categories : 논문리뷰
tags :
- 추천알고리즘
- 논문리뷰
---
수정중

[논문 링크](https://dl.acm.org/doi/abs/10.1145/2959100.2959190/)

유투브는 사용자의 이전 기록을 이용하여 사용자 맞춤으로 동영상을 추천해준다. 

# ABSTRACT
1. deep candidate generation model
2. separate deep ranking model

**#추천시스템; #딥러닝; #확장성**



# 1. INTRODUCTION

유투브는 전 세계에서 가장 많은 사용자들이 이용하는 비디오 공유 플랫폼이다. 20억명이 넘는 사용자들이 사용하고 끊임없이 동영상이 업로드 되고 있다. 따라서 유투브 추천 알고리즘은 세가지 관점에서 도전적인 과제라고 볼 수 있다.

1. 규모(Scale) : 작은 규모에서 작동했던 추천 알고리즘은 유투브에 적용하면 작동하지 않았다. 유투브에 특화된 알고리즘이 필요할 뿐만 아니라 효율적인 서버 시스템이 필요하다.
2. (Freshness) : 유투브는 끊임없이 동영상이 업로드 되기 때문에 코퍼스가 일정하지 않다.
3. 잡음(Noise) : 사용자가 시청한 동영상은 전체 동영상의 극히 일부(sparsity)이고, 사용자가 시청한 동영상을 마음에 들어하는지 알 수 없다.

추천 시스템의 이전 연구들은 대부분 matrix-factorization을 사용하고 딥러닝을 이용한 연구는 상대적으로 적다. 

논문의 전개 과정은 다음과 같다.
  - Section2) 시스템 overview
  - Section3) candidate generation model
  - Section4) ranking model
  - Section5) 결론



# 2. SYSTEM OVERVIEW

- 후보 생성 네트워크 (The candidate generation network)

  사용자의 시청 기록을 인풋으로 넣고, 큰 코퍼스에서 작은 서브셋을 검색한다. 후보 생성 네트워크는 협동 필터링(collaborative filtering)을 이용한 광범위한 개인화만을 제공한다. 사용자 간의 유사성은 coarse features 관점에서 표현된다. 여기서 말하는 coarse features는 비디오 시청한 ID, 검색 쿼리 토큰, 인구통계정보를 의미한다.

- 랭킹 네트워크 (The ranking network)
  
  후보 중에 상대적인 중요도를 구분하기 위해서 세밀한 표현이 필요하다. 여기에는 재현율(recall, 실제 True 중 True로 예측한 비율)이 사용된다. 
  
  
모델 성능은 두가지 방법으로 측정한다. 
  1. Offline Experiments : precision, recall, ranking loss
  2. Live Experiments : 클릭률, 시청시간   

두 실험의 결과가 항상 상관관계가 있진 않다. 메트릭으로 나타나지 않는 실제 결과를 A/B 테스트를 통해 알아보고자 두가지 실험을 병행한다.




# 3. CANDIDATE GENERATION

이 모델의 전신은 rank loss로 학습된 matrix factorization 접근법이다. 신경망 모델의 초기 반복은 사용자의 이전 시청기록만 포함하는 얕은 네트워크(shallow networks)로 이러한 분해(factorization)을 모방한다. 이런 관점에서 우리의 접근법이 factorization의 비선형 일반화라고 볼 수 있다.


## 3.1 Recommendation as Classification

극단적인 다중 분류를 하여 추천한다. 

```math
P(w_t=i|U,C) = \freq(e^{v_iu},\Sigma_{j\inV}e^{v_ju} 
```
