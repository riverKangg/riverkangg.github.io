---
title : "[Study Scala] 함수"
date : 2022-06-06
categories : Scala

tags :
- Scala
- 코딩공부
---
## 프로시저
- 반환값을 가지지 않는 함수
- ``` def pnt(s: String) = println(s"Print input: $s") ```

## 재귀함수
- 재귀적 호출이 추가적인 스택공간을 사용하지 않는 꼬리-재귀(tail-recursion) 가능
- ```    
    @annotation.tailrec
    def power(x: Int, n: Int, x: Int = 1): Int = {
       if (n < 1) t
       else power(x, n-1, x*t)
