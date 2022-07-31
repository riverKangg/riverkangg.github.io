---
title : "[Study Scala] 표현식과 조건문"
date : 2022-06-05
categories : Scala

tags :

- Scala
- 코딩공부

---

# 표현식

- 표현식
    - <code> val x = 4 * 5; val amt = x - 3 </code>
- 표현식 블록
    - <code> val amt = { val x = 4 * 5; x - 3 } </code>

### 1. If-Else 표현식

- <code> val min = if (x < y) x else y </code>

### 2. 매치 표현식

- If-Else 표현식보다 선호
- ```
      val x = 1; val x = 2;
      val min = x < y match {  
                     case true => x   
                     case false => y
        }
    ```

#### 2-1 와일드 카드 매칭

- 매치 표현식에서 입력되지 않은 패턴이 나왔을 때 대비
- ```
      val message = "Yes"
      val answer = message match {
            case "Yes" => "OK"
            case _ => {
              println(s"I don't know this answer : $message")
              "NO"
            }
      }
  ```

#### 2-2 패턴 가드 매칭

- 값 바인딩 패던에 if 표현식 추가

#### 2-3 패턴 변수를 사용한 타입 매칭

- ```
      val check: Int = 1
      check match {
          case x: String => s"'x'"
          case x: Double => f"$x%.2f"
          case x: Long => s"${x}l"
          case x: Int => s"${x}i"
       }
  ```

### 3. 루프

#### for

- ``` for (x <- 1 to 3) { println(s"Count $x") } ```

#### for : 값 바인딩

- ``` val powerOf2 = for (i <- 0 to 3; pow = 1 << i) yield pow ```

#### While, Do/While

- ``` val x = 3; while (x < 10) x += 1; ```
- ``` val x = 3; do println(s"under 10, x = $x") while (x < 10)```            
