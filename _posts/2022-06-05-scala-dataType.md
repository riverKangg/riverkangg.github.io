---
title : "[Study Scala] 데이터 타입"
date : 2022-06-05
categories : Scala

tags :
- Scala
- 코딩공부
---

# 1. 스칼라 데이터 타입

### 1. literal(리터럴)
- 소스 코드에 바로 등장하는 데이터
- _리터럴 숫자 5 할당_
  - <code lang="scala">
      val x: Int = 5
    </code>

### 2. value(값)
- 불변의 타입을 갖는 저장 단위
- 정의할 때만 할당
- 재할당 불가
- _타입 추론 가능_
  - <code> val x = 20 </code>

### 3. variable(변수)
- 가변의 타입을 갖는 저장 단위
- 정의할 때 할당
- 재할당 가능
- _타입 추론 가능_
  - <code> var x = 20 </code>

### 4. type(타입)
- 계층구조
  - [스칼라 공식 링크](https://docs.scala-lang.org/ko/tour/unified-types.html)
  - ![image](https://user-images.githubusercontent.com/45582326/172026611-67f1493c-87ff-42da-b020-87cdde87aee0.png)
- 튜플
  - 둘 이상의 값을 가지는 순서가 있는 컨테이너
  - 다른 타입의 값 가능
  - <code> val tup = (1, true, "Sun")
