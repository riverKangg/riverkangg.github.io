---
title : "[Python/프로그래머스] 동적계획법-정수삼각형"
date : 2020-08-30
categroies : python
tags :
- 프로그래머스
published : false
---

동적계획법(Dynamic Programming)-lv3.정수삼각형


## ▷ 문제설명
[문제 링크]( https://programmers.co.kr/learn/courses/30/lessons/43105 )



## ▷ 아이디어
1. i번째 층은 i+1번째 층에 더해진다.
2. i+1번째 층의 양 끝은 경우의 수가 한개뿐이다.
3. i+1번째 층의 양 끝을 제외한 가운데 수는 두가지 경우의 수 중 큰 수를 선택한다.



## ▷ 풀이코드
```{Python}

def solution(n, lost, reserve):
    
    def solution(triangle):

    for i in range(len(triangle)-1):
        tri1,tri2 = triangle[i],triangle[i+1]
        temp=[]
        
        for idx,j in enumerate(tri2):
            if idx==0:
                temp.append(j+tri1[idx])
            elif idx<len(tri2)-1:
                temp.append( max(j+tri1[idx-1],j+tri1[idx]) )
            else:
                temp.append( j+tri1[idx-1] )
            triangle[i+1]=temp
            
    answer = max(triangle[-1])
    
    return answer
    
```


## ▷ 다른풀이
```{Python}
solution = lambda t, l = []: max(l) if not t else solution(t[1:], [max(x,y)+z for x,y,z in zip([0]+l, l+[0], t[0])])
```
