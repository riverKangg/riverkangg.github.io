---
titles : "금융 데이터에 CNN 적용해보기"
data : 2020-01-01
published : false
---




은행 데이터는 구조화(structured)되어 있어서(2차원/3차원의 테이블) 딥러닝이 필요없다고 볼 수 있지만 적용해본 결과 잘 작동했다는 글을 봤다.


## 글 요약
[Convolutional Neural Network on a structured bank customer data](https://towardsdatascience.com/convolutional-neural-network-on-a-structured-bank-customer-data-358e6b8aa759)
### The problem
- 캐글의 은행상품추천 대회에 적용해본 모델이다. [캐글링크](https://www.kaggle.com/c/santander-product-recommendation)
- 데이터는 고객의 금융상태를 월단위 스냅샷으로 찍은 것과 인구통계정보(성별, 나이, 위치, 소득 등)이다.
- 은행에서 제공하는 상품은 수표, 저축, 모기지, 신용 카드 등 24 개가 있다.
- 2015.01-2016.05의 고객 제품 사용량과 2015.01-2016.06의 기타 정보를 기반으로 2016.06의 24 개의 상품 사용량을 예측하는 문제다.(0에서 1로 변경된 상품만 고려)
- 고객이 특정 월에 상품을 사용하면 1 아니면 0으로 표시한다. 고객이 특정 달에 여러 상품을 사용할 수 있기 때문에 multi-class multi-label classification이다.
- 전형적인 패널데이터다. 타겟이 default label로만 바뀌면, behavior scorecard model을 위한 전형적인 데이터 셋이다. 
- 하지만 이를 시간과 공간(상품 사용)이 있다고 볼수도 있다.

### CNN’s power — Feature Engineering
- 캐글에서는 feature engineering이 중요한데, CNN을 사용해서 이 단계를 줄인다. 
- 고객데이터를 이미지 데이터로 바꾼다. 모든 고객은 은행과 거래를 시작한 시점부터 현재까지 시간차원이 있다. 소득이나 은행과의 관계 같은 피처들은 시간에 따라서 바뀐다. 
이런 피처들까지 고려해서 최종 25x17 픽셀 이미지를 생성한다. 

### The model
- Keras 사용(Tensorflow 벡엔드)
- two path
  - convolution/pooling/dropout -> densely connected Neural Network 
  - plain densely connected Neural
- 몇개의 피처는 시간에 따라 변하지 않기 때문
- layers, nodes 조정에 많은 시간을 들였고,
- dropout rates, L1 and L2 penalty 는 조정하지 않음

### Final thoughts
- CNN의 컨볼루션이나 풀링 필터 내에서 time dependency 정보가 무시된다는 약점있다.
- 하지만 CNN이 RNN보다 학습시키기 쉽다.



## 내 생각
- 금융 데이터는 RNN이 더 작합할 것 같지만, CNN도 시도해볼만한다.
