##############
###  설정  ###
##############

# 결과 저장 경로
save_path = ""

# 카페 정보
cafe = {'name':                          # 카페 이름
       ,'page_link':                     # 주소
cafe.update({"keywords" : [""]})         # 검색 키워드





###############################
###  카페 게시글 링크 수집  ###
###############################

for keyword in cafe.get("keywords"):
    ### 카페 주소 입력
    driver.get(cafe.get("page_link"))
    
    ### 키워드 검색
    clipboard_input('//*[@id="topLayerQueryInput"]', keyword)
    try :
        driver.find_element_by_xpath('//*[@id="cafe-search"]/form/button').click()   # 왼쪽에 검색창
    except:
        driver.find_element_by_xpath('//*[@id="info-search"]/form/button').click()   # 오른쪽에 검색창
    driver.implicitly_wait(0.5)
    driver.switch_to.frame('cafe_main')

    ### 키워드 수집 정보
    num_per_page = 15          # 페이지당 게시글 갯수(default: 15개)

    address_list=[]
    page = 1
    
    l=True
    while l:
        
        time.sleep( random.randint(0,5) )
        
        ### 현재 페이지의 html 불러오기
        r = driver.page_source
        page_html = BeautifulSoup(r, "html.parser")
        content = page_html.find("div", class_="article-board result-board m-tcol-c").find('tbody')
#         content = page_html.find_all("div", class_="article-board m-tcol-c")[1].find('tbody')
        body = content.find_all("tr")

        ### 게시글 정보 저장하기
        for x in body:
            temp_dict={}
            if x.find("div", class_="board-number") is not None:
                temp_dict['no'] = x.find("div", class_="board-number").text.strip()
                temp_dict['title'] = x.find("div", class_="board-list").text.strip().replace('  ','').replace('\n','')
                temp_dict['link'] = x.find('a').get('href')
                temp_dict['name'] = x.find("td", class_="td_name").find('a',class_='m-tcol-c').text.strip()
                temp_dict['date'] = x.find("td", class_="td_date").text.strip()
                temp_dict['view'] = x.find("td", class_="td_view").text.strip()
                address_list.append(temp_dict)
        print("(현재시각) "+str(datetime.datetime.now())+": "+ str(page) +"page done")

        ### 다음 페이지로 넘어가기
        page+=1
        driver.implicitly_wait(1)
        try:
            if page<=10:   # 1~10 : 페이지 번호 그대로
                page_xpath = str(page)
                driver.find_element_by_xpath('//*[@id="main-area"]/div[7]/a[' + page_xpath + ']').click()
            elif page == 11:   # 11 : 다음 버튼
                driver.find_element_by_xpath('//*[@id="main-area"]/div[7]/a[11]/span').click()
            elif page>11 and page%10!=1:   # 12~ : 페이지 번호 마지막 자리 + 1
                page_xpath = str(page-((page-1)//10)*10+1)
                driver.find_element_by_xpath('//*[@id="main-area"]/div[7]/a[' + page_xpath + ']').click()
            elif page%10 == 1:   # 21,31.. : 다음 버튼
                driver.find_element_by_xpath('//*[@id="main-area"]/div[7]/a[12]/span').click()
        except:
                address_df = pd.DataFrame(address_list)
                address_df['idx_no'] = range(1,len(address_df)+1)   # 조인할 키 값
                address_df.to_pickle(save_path+"cafe_address_"+cafe.get("name")+"_"+keyword+".pkl")
                print("(현재시각) "+str(datetime.datetime.now())+": done")
                l=False
if len(set(address_df['no']))!=len(address_df) :
    print("게시글 번호에 중복 존재")
print("검색게시글수 : ", address_df.shape)
display(address_df.head())





################################
###  카페 게시글 내용 수집   ###
################################
import pickle
from contextlib import suppress

for keyword in cafe.get("keywords"):
    df = pickle.load(open(save_path+"cafe_address_"+cafe.get("name")+"_"+keyword+".pkl", 'rb'))

    i=0
    contents_list = []   # 내용
    reply_list = []      # 댓글
    error_list = []      # 에러난 게시글

    while True:

        ### 수집 링크로 이동
        url = "https://cafe.naver.com"+df.loc[i,'link']
        idx_no = df.loc[i,'idx_no']    # 인덱스 번호
        driver.get(url)
        time.sleep( random.randint(2,5) )
        try:
            driver.switch_to.frame('cafe_main')
            time.sleep( random.randint(2,5) )
            r = driver.page_source
            page_soup = BeautifulSoup(r, "html.parser")
            content = page_soup.find('div', class_='ArticleContentBox')  

            ### 게시글 수집
            temp_dict={}
            temp_dict['idx_no'] = idx_no
            temp_dict['title'] = ""
            with suppress(AttributeError):   # 제목 없는 게시글
                temp_dict['title'] = content.find('h3',class_='title_text').text.strip()
            temp_dict['content'] = content.find("div", class_="article_viewer").text.strip()
            temp_dict['nick'] = content.find('div',class_='profile_info').find('a',class_='nickname').text.strip()
            temp_dict['date'] = content.find('div',class_='article_info').find('span',class_='date').text.strip()
            temp_dict['view'] = ""
            with suppress(AttributeError):
                temp_dict['view'] = content.find('div',class_='article_info').find('span',class_='count').text.strip()
            contents_list.append(temp_dict)

            ### 댓글 수집
            if content.find("div", class_="ReplyBox") is not None:   # 댓글 기능이 아예 없음  
                comment_num = content.find("div", class_="ReplyBox").find("a",class_="button_comment").find("strong").text
                if comment_num!='0':   # 댓글이 없음
                    comment = content.find("div", class_="CommentBox").find("ul",class_="comment_list").select("li")
                    
                    ### 댓글 구분
                    com_n=0    # 댓글
                    com_nn=0   # 대댓글
                    
                    for n in range(len(comment)):

                        if comment[n].get('class')==['CommentItem']:    # 댓글
                            com_n+=1; com_nn=0;
                            com_thread = str(com_n)+"-"+str(com_nn)
                            com_nn=1
                        elif comment[n].get('class')==['CommentItem', 'CommentItem--reply']:    # 대댓글
                            com_thread = str(com_n)+"-"+str(com_nn)
                            com_nn+=1

                        ### 댓글 내용 수집    
                        if comment[n].text.strip() != '삭제된 댓글입니다.':
                            com_nick = comment[n].find("a",class_="comment_nickname").text.strip()
                            com_date = comment[n].find("span",class_="comment_info_date").text.strip()
                            com_reply = comment[n].find("div",class_="comment_text_box").text.strip()
                            reply_list.append({'idx_no':idx_no, 'nick':com_nick, 'date':com_date, 'reply':com_reply, "thread":com_thread})
            i+=1

        except:
            i+=1
            ### 게시글을 볼 등급이 안됨
            if page_soup.find('strong', class_='emph') is not None:
                error_list.append({"error" :  page_soup.find('strong', class_='emph').text+"등급 필요"
                                   , "url" : url})
                pass
            ### 에러 따로 확인
            else:
                error_list.append({"error" : "에러 확인 필요"
                                   , "url" : url})
                pass

        ### 수집한 글 갯수만큼 반복
        if i == len(df):
            contents_df = pd.DataFrame(contents_list)
            contents_df.to_pickle("../../data/cafe/cafe_contents_"+cafe.get("name")+"_"+keyword+".pkl")
            reply_df = pd.DataFrame(reply_list)
            reply_df.to_pickle("../../data/cafe/cafe_replies_"+cafe.get("name")+"_"+keyword+".pkl")
            print("(현재시각) "+str(datetime.datetime.now())+": done")
            break
    
# 크롬 종료 
driver.quit()

# 수집한 데이터 : contents_df
print("수집 데이터 : ", contents_df.shape)
# 에러 난 게시글 : error_list
print("에러 게시글 수 : ", len(error_list))
