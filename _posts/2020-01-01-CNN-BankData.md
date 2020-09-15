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
- 패널데이터
- behavior scorecard model ????

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
