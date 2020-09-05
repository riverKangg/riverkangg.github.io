---
title: "[Python/Crawling] 네이버 로그인(캡챠 해결)"
date: 2020-09-04
categories: 크롤링
tags : 
- 크롤링
- crawling
---

수정중

네이버 로그인이 가능한 크롤러를 만들기 위해서 크롬 드라이버를 설치되어 있어야한다. 설치되어 있지 않다면 다음과 같은 방법으로 설치해보자.

# ▷ 크롬 드라이버 설치
1. 내 크롬 버전 확인    
<img src="https://github.com/riverKangg/riverkangg.github.io/blob/master/_posts/image/2020-09-04-crawling-%ED%81%AC%EB%A1%AC%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B2%84%ED%99%95%EC%9D%B8.png" width="600px" title="px(픽셀) 크기 설정" alt=""></img><br/>

2. 크롬 버전에 맞는 드라이버 다운로드    
[https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)

3. 압축 풀고 원하는 경로로 **chromedriver.exe**를 이동




# ▷ 네이버 로그인 코드
### □ 라이브러리 호출
```{Python}
import time
import pyperclip
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
```
### □ 
```{Python}
#######################
###  네이버 로그인  ###
#######################

# 크롬 웹 드라이버의 경로를 설정
driverLoc = # 크롬 드라이버 경로 입력
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(driverLoc, chrome_options=options)

# 네이버 로그인 페이지 접속
driver.get("https://nid.naver.com/nidlogin.login")

### 로그인 정보
login = {"id" : ""   # 네이버 아이디
        ,"pw" : "" # 네이버 비밀번호
        }

# 로그인 정보 입력 함수
def clipboard_input(user_xpath, user_input):
    temp_user_input = pyperclip.paste()  # 사용자 클립보드를 따로 저장

    pyperclip.copy(user_input)
    driver.find_element_by_xpath(user_xpath).click()
    ActionChains(driver).key_down(Keys.CONTROL).send_keys('v').key_up(Keys.CONTROL).perform()

    pyperclip.copy(temp_user_input)  # 사용자 클립보드에 저장 된 내용을 다시 가져 옴
    time.sleep(1)


# id, pw 입력 후 클릭
clipboard_input('//*[@id="id"]', login.get("id"))
clipboard_input('//*[@id="pw"]', login.get("pw"))
driver.find_element_by_xpath('//*[@id="log.login"]').click()

time.sleep(10)
```
# ▷ 코드 다운로드
[코드 다운로드 링크](https://github.com/riverKangg/riverkangg.github.io/blob/master/code/2020-09-04-crawling-1login.py/)
