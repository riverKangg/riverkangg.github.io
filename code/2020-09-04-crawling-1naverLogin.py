import time
import pyperclip
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

#######################
###  네이버 로그인  ###
#######################

# 크롬 웹 드라이버의 경로를 설정
driverLoc = # 크롬 드라이버 경로 입력
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # 브라우저 숨기기
driver = webdriver.Chrome(driverLoc, chrome_options=options)

# 네이버 로그인 페이지 접속
driver.get("https://nid.naver.com/nidlogin.login")

# 로그인 정보
login = {"id" : ""   # 네이버 아이디
        ,"pw" : ""   # 네이버 비밀번호
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
