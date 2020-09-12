---
title : "[논문리뷰] Deep Neural Networks for YouTube Recommendations"
date : 2020-09-08
categories : 논문리뷰
tags :
- 추천알고리즘
- 논문리뷰
use_math: true
---
수정중


2016년 구글 리서치가 공개한 논문이다. 이 논문에서는 YouTube라는 영상 플랫폼의 특징을 고려한 추천시스템을 설명하고 있다.

[논문 링크](http://research.google.com/pubs/pub45530.html?utm_content=bufferf6bbc&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer)

# ABSTRACT
1. deep candidate generation model
2. separate deep ranking model

**#추천시스템; #딥러닝; #확장성**



# 1. INTRODUCTION

Youtube 추천시스템은 세가지 관점을 고려해서 만들었다.

1. 규모(Scale) : 작은 규모에서 작동했던 추천 알고리즘은 유투브에 적용하면 작동하지 않았다. 유투브에 특화된 알고리즘이 필요할 뿐만 아니라 효율적인 서버 시스템이 필요하다.
2. 새로움(Freshness) : 유투브는 끊임없이 동영상이 업로드 되기 때문에 후보군(코퍼스)이 일정하지 않다.
3. 잡음(Noise) : 사용자가 시청한 동영상은 전체 동영상의 극히 일부(sparsity)이고, 사용자가 시청한 동영상을 마음에 들어하는지 정확한 피드백이 없다.

이전 연구들은 대부분 matrix-factorization을 사용하고 딥러닝을 이용한 연구는 상대적으로 적었다. 



# 2. SYSTEM OVERVIEW

<p align="center">
  <img src="https://raw.githubusercontent.com/riverKangg/riverkangg.github.io/master/_posts/image/2020-09-10-fig2.png" width=500>
</p>

위 그림이 추천시스템의 전체적인 구성이고, 파란색 블럭이 실제 추천을 진행하는 단계이다.

- 후보 생성 네트워크 (The candidate generation network)

    - 협동 필터링(collaborative filtering)으로 넓은 의미의 개인화를 제공한다.    
    - 사용자 간의 유사성은 coarse features 관점에서 표현된다. 여기서 말하는 coarse features는 비디오 시청한 ID, 검색 쿼리 토큰, 인구통계정보를 의미한다.
    
- 랭킹 네트워크 (The ranking network)
    
    - 상대적인 중요도를 구분하여 세밀한 추천 목록 생성한다. 
    - 여기에는 재현율(recall, 실제 True 중 True로 예측한 비율)이 사용된다. 
    
이 과정에서 사용자의 시청기록과 맥락을 고려한다.

  
  
모델 성능은 두가지 방법으로 측정한다. 
  1. Offline Experiments : precision, recall, ranking loss
  2. Live Experiments : 클릭률, 시청시간   

두 실험의 결과가 항상 똑같진 않다. 1번 방법의 메트릭으로 나타나지 않는 실제 결과를 A/B 테스트를 통해 알아보고자 2번째 방법을 병행한다.




# 3. CANDIDATE GENERATION

## 3.1 Recommendation as Classification

극단적인 다중 분류를 하여 추천한다. 

$$ K(a,b) = \int \mathcal{D}x(t) \exp(2\pi i S[x]/\hbar) $$


### *Efficient Extreme Multiclass*

실제 레이블과 샘플링 된 네거티브 클래스에 대해 교차 엔트로피 손실이 최소화 된다.

사용자에게 보여줄 top N개를 뽑는다. 수백만개의 항목들



## 3.2 Model Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/riverKangg/riverkangg.github.io/master/_posts/image/2020-09-10-fig3.png" width=500>
</p>

맨 아래 입력부터 맨 위 아웃풋까지의 시스템을 나타낸다. 이 그림에서 시청기록은 파란색을 나타낸다. 이 부분을 자세하게 살펴보자.

1. 각 영상마다 고정된 길이로 *고차원 임베딩*을 학습시킨다.
2. 이 임베딩을 *피드포워드 신경망*에 넣는다. 사용자의 시청 기록은 sparse한 영상 ID에 대한 가변길이 시퀀스로 나타낸다. 이 시퀀스는 임베딩으로 된 dense 벡터 표현으로 매핑된 것이다. 
네트워크는 고정길이 dense를 입력이 필요하고 여러 방법(합, 구성요소별 최대 등) 중에서 가장 잘 수행 된 임베딩의 평균을 구한다. 
3. 피처들은 넓은 첫번째 레이어와 연될되고, 그 다음에는 여러 개의 완전 연결된 ReLU 레이어로 이어진다.

임베딩 된 sparse 피처를 보여주는 심층후보생성모델구조는 dense 피처로 연결된다. 
임베딩은 연결 전에 평균화 되어 히든레이어에 대한 입력에 적합한 가변 크기의 sparse ID 백을 고정 너비 벡터로 변환한다.
모든 히든레이어는 완전연결(fully connected)되어 있다. 훈련하면서 생플링 된 소프트맥스 출력에서 경사하강법으로 cross-entropy loss가 최소화된다. 
실시간으로 수백개의 후보 영상 추천을 생성하기위해 대략적인 최근접이웃 조회가 수행된다.


## 3.3 Heterogeneous Signals

- **검색기록**도 비슷한 방법으로 입력한다. - 각 검색어를 unigram이나 bigram으로 토큰화하고, 이 토큰을 임베딩한다. 검색기록은 그림에서 초록색으로 표현되어 있다.
- **인구통계학적 피처**
    - 사용자의 위치, 기기정보는 임베딩되어 합쳐진다.
    - 간단한 이진피처나 연속형 피처(성별, 로그인 상태, 나이)는 0과 1 사이의 값으로 표준화해서 입력한다.


### "Example Age" Feature
<p align="center">
  <img src="https://raw.githubusercontent.com/riverKangg/riverkangg.github.io/master/_posts/image/2020-09-10-fig4.png" width=400>
</p>
영상의 나이를 피처로 학습시켰을 때, 정확한 표현이 가능하다. 그래프를 보면 영상나이를 넣지 않은 baseline모델(파란색)은 training window내의 평균 가능성으로만 예측한다. 

## 3.4 Label and Context Selection 
- 훈련예제는 모든 YouTube 시청 데이터로 생성 
    - 
    - 사용자가 추천 이외의 방법으로 영상을 찾을 때, 이 결과를 빠르게 전달한다.
- 사용자 별로 고정된 수의 훈련예제를 생성 
    - loss function에 모든 사용자들이 동일한 가중치를 가져가도록 한다.
    - 이는 매우 활발한 소수의 사용자에게 집중되어 추천시스템이 만들어지는 것을 방지한다.


## 3.5 Experiments with Features and Depth



# 4. RANKING
<p align="center">
  <img src="https://raw.githubusercontent.com/riverKangg/riverkangg.github.io/master/_posts/image/2020-09-10-fig7.png" width=500>
</p>

## 4.1 Feature Representation

### *Feature Engineering*

### *Embedding Categorical Features*

### *Normalizing Continuous Features*

## 4.2 Modeling Expected Watch Time

## 4.3 Experiments with Hidden Layers



# 5. CONCLUSIONS

# 6. ACKNOWLEDGMENTS
