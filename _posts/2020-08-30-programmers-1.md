---
title : "[프로그래머스] 동적계획법lv3-정수삼각형"
date : 2020-08-30
categroies : 프로그래머스
tags :
- 코딩
- 프로그래머스 풀이
- programmers
---

동적계획법(Dynamic Programming)-lv3.정수삼각형


## ▷ 문제설명
[문제 링크]( https://programmers.co.kr/learn/courses/30/lessons/43105 )



## ▷ 아이디어
> 1. 



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
