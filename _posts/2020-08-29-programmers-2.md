---
title: "[프로그래머스] 탐욕법lv2-큰 수 만들기"
date: 2020-08-29 08:26:28 -0400
categories: 프로그래머스
---

탐욕법(Greedy)-lv2.큰 수 만들기

탐욕 알고리즘은 

탐욕 알고리즘에 대한 더 자세한 설명은 다음 글을 참고하자. \
[탐욕 알고리즘]()

[lv1.체육복](https://riverkangg.github.io/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4/programmers-1/)



## ▷ 문제설명
[문제 링크]( https://programmers.co.kr/learn/courses/30/lessons/42862)    



## ▷ 아이디어
이 문제는 시간초과 때문에 힘들었다. 세번이나 완전 새로운 로직을 짠 후에나 통과할 수 있었다.
1. max 함수, string을 리스트로 바꾸는 방법은 연산량이 많아 사용하지 않는 것이 좋다.
2. i번째 숫자보다 i+1번째 숫자가 더 크면 i번째 숫자를 없앤다.
3. 없애는 숫자의 갯수인 k가 0이 되면 중단한다. 
4. i+1번째 숫자부터 마지막 숫자까지의 갯수가 k라면 알고리즘을 중단한다.



## ▷ 풀이코드
```{Python}
def solution(number, k):

    i=0
    while k>0 and i+1<len(number):

        if number[i] < number[i+1] :
            number = number[:i] + number[i+1:]
            k-=1
            if i>0:
                i-=1
        else:
            i+=1

        if i+1==len(number) and k>0:
            return number[:-k]
        elif k==0:
            return number

    return(number)
```



## ▷ 다른 풀이

