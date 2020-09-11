---
title : "[Python/모듈] functools-lru_cache"
date : 2020-08-30
categories : python
tags : python모듈
---


이번 포스팅에서는 DP문제를 푸는데 lru_cache를 소개하고자 한다. "LRU(least recently used) 캐시"는 

functools는 고차 함수를 위한 모듈이다.
모듈 설명은 [functools](https://python.flowdas.com/library/functools.html)를 참고하자.

## ▷ 예시 코드
```{Python}
from functools import lru_cache
```


## ▷ 예시 문제

lru_cache를 사용하면 동적프로그래밍을 효과적으로 할 수 있다. lru_cache를 사용해서 다음 문제를 풀어보자.

[프로그래머스 lv3-등굣길](https://programmers.co.kr/learn/courses/30/lessons/42898)
