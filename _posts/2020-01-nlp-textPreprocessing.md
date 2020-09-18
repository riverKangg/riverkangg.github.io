---
titles : "한국어 텍스트 전처리"
data : 2020-01-01
published : false
---

자연어처리를 해본 결과 스텝을 정형화 할 수 없음을 뼈져리게 느꼈다. 데이터마다 필요한 전처리 과정이 달랐기 때문이다. 

1. Tokenization
2. Stop Words Removal
3. Morphological Normalization
4. Collocation


## 1. Tokenization
영어에 비해 특히 어려운 전처리가 토큰화 과정이다. 
- 영어는 공백을 나누면 단어로 떨어지는 언어이고, 한국어는 공백을 나눈다고 해서 단어로 깔끔하게 나눠지지 않는다. (이걸 표현하는 단어가 있나?)

### Reference
[1] <https://towardsdatascience.com/building-blocks-text-pre-processing-641cae8ba3bf>
