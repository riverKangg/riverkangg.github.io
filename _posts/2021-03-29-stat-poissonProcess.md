---
title : "Poisson Process 시뮬레이션"
date : 2021-03-29
categories : Python
published : False
tags :
- 추천알고리즘
- 기술리뷰
---

# Poisson Distribution(포아송 분포)
포아송 과정을 말하기 전에 포아송 분포를 먼저 살펴보겠습니다. 포아송 분포는 단위 시간 안에 특정 사건이 몇 번 일어날지를 나타냅니다. 일정 시간동안 도착하는 손님 수로 자주 설명하고 있죠. 단위 시간 안에 도착하는 손님의 평균이 **&lambda;** 일 때, 그 사건이 **k**회 일어날 횟수를 식으로 나타내면 다음과 같습니다.
여기서 x는 확률 변수를 나타내고, 


# Poisson Process(포아송 과정)
포아송 과정은 포아송 분포의 독립적인 결합으로 만들어집니다. 포아송 과정의 정의는 다음과 같습니다.
> (1) N(0)=0
> (2) 독립 증분(Independent Increment) 
> (3) 정상 증분(Stationary Increments) 이 식을 풀어서 해석해보면, 도착 시간의 분포는 오직 구간의 길이에만 영향을 받는다는 의미입니다. 예를 들면, 1시부터 1시간 이내에 도착한 손님의 분포나 4시부터 1시간 동안 도착한 손님의 분포는 동일해야 한다는거죠.(하지만 현실은 그렇지 않죠!)
> (4) No counted occurrence are simultaneous

### Reference
**[1]** https://towardsdatascience.com/the-poisson-process-everything-you-need-to-know-322aa0ab9e9a </br>
**[]** https://fromosia.wordpress.com/2017/03/19/stochastic-poisson-process/ </br>
**[2]** https://en.wikipedia.org/wiki/Poisson_distribution   </br>

