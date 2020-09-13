---
title: "[Python/프로그래머스] 탐욕법-큰 수 만들기"
date: 2020-08-29 08:26:28 -0400
categories: python
tags : 프로그래머스
publised : false
---

탐욕법(Greedy)-lv2.큰 수 만들기


## ▷ 문제설명

number에서 k개의 숫자를 빼서 가장 큰 수를 만드는 문제이다. 자세한 문제설명은 다음 링크를 참고하자.

<https://programmers.co.kr/learn/courses/30/lessons/42862/>

특정 만 통과가 안된다면 다음 테스트 케이스를 추가로 시도해보자.

- 테스트 케이스
    - number, k = '87654321', 3
    - number, k = '9'*1000000,999999
    - number, k = '10001',2



## ▷ 아이디어
이 문제는 시간초과 때문에 힘들었다. 세번이나 완전 새로운 로직을 짠 후에나 통과할 수 있었다.
1. max 함수, string을 리스트로 바꾸는 방법은 연산량이 많아 사용하지 않는 것이 좋다.
2. i번째 숫자보다 i+1번째 숫자가 더 크면 i번째 숫자를 없앤다.
3. 없애는 숫자의 갯수인 k가 0이 되면 중단한다. 
4. i+1번째 숫자부터 마지막 숫자까지의 갯수가 k라면 알고리즘을 중단한다.



## ▷ 풀이코드

#### 첫번째 시도

나름 간단하게 짰다고 생각했는데 10번 테스트를 통과하지 못했다. max 함수와 index 함수가 문제인듯하다. 

```{Python}
def solution(number, k):

    answer=''
    while k>0 :
            
        num = max(number[:k+1])
        idx = number.index(num)

        answer+=num
        l = len(number)
        
        if l>k :
            number = number[idx+1:]
            k-=idx
            l-=idx+1
            
        if k==0 :
            answer+=number
            return answer
        
        elif k==l :
            return answer
```



#### 두번째 시도

두번째 시도에서는 i번째와 i+1번째 수를 비교하는 방법을 사용했다. 전체 수를 탐색하지 않아서 연산량을 많이 줄일 수 있었다.

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

