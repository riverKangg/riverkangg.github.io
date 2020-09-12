#-*- coding: utf-8 -*-
import os
import sys
import urllib.request

### 네이버API id, pw 입력
client_id =     # 네이버API id
client_secret = # 네이버API pw

### 데이터랩 API 주소
url = "https://openapi.naver.com/v1/datalab/search";

### 검색에 사용되는 조건
body_temp = {'startDate': '2017-01-01'
             , 'endDate': '2017-04-30'
             , 'timeUnit': 'month'     # input : [date, week, month]
        
             ### 그룹, 키워드 입력
             , 'keywordGroups': [{'groupName': '그룹1', 'keywords': ['word1', 'word2']}
                                 , {'groupName': '그룹1', 'keywords': ['word1', 'word2']}]
             , 'device': 'pc'
             , 'ages': ['1', '2']
             , 'gender': 'f'
            }

### 딕셔너리를 str 형식으로 변환
body = str(body_temp).replace("'",'"')

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
request.add_header("Content-Type","application/json")
response = urllib.request.urlopen(request, data=body.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    ### str 형식으로 반환됨
    response_body = response.read().decode('utf-8')
    print(response_body)
else:
    print("Error Code:" + rescode)
    
### 딕셔너리 형식으로 변환
dic = eval(response_body)
dic
