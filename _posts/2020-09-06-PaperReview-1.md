
---
title : "[논문리뷰] Deep Neural Networks for YouTube Recommendations"
date : 2020.09.06
categories : 논문리뷰
tags :
- 추천알고리즘
- 논문리뷰
---
수정중

유투브는 사용자의 이전 기록을 이용하여 사용자 맞춤으로 동영상을 추천해준다. 

# ABSTRACT
논문의 전개과정
1. deep candidate generation model
2. separate deep ranking model

**#추천시스템; #딥러닝; #확장성**

# 1. INTRODUCTION
유투브는 전 세계에서 가장 많은 사용자들이 이용하는 비디오 공유 플랫폼이다. 20억명이 넘는 사용자들이 사용하고 끊임없이 동영상이 업로드 되고 있다. 따라서 유투브 추천 알고리즘은 세가지 관점에서 도전적인 과제라고 볼 수 있다.

1. 규모(Scale) : 작은 규모에서 작동했던 추천 알고리즘은 유투브에 적용하면 작동하지 않았다. 유투브에 특화된 알고리즘이 필요할 뿐만 아니라 효율적인 서버 시스템이 필요하다.
2. (Freshness) : 유투브는 끊임없이 동영상이 업로드 되기 때문에 코퍼스가 일정하지 않다.
3. 잡음(Noise) : 
