import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from konlpy.tag import Okt
import ast
from sklearn.metrics.pairwise import cosine_similarity
import hashlib  # 입력 해시 생성용
import os
import json
import time

# MPS 장치 사용 여부 확인
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# KoBERT 모델과 토크나이저 로드 및 초기화
model_name = 'monologg/kobert'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.to(device)
print("KoBERT 모델과 토크나이저 로드 완료")

# 단어 임베딩을 캐싱하기 위한 딕셔너리
embedding_cache = {}

# 캐시 디렉토리 설정
CACHE_DIR = './cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_embeddings_with_cache(words):
    embeddings = []
    words_to_process = []
    for word in words:
        if word in embedding_cache:
            embeddings.append(embedding_cache[word])
        else:
            words_to_process.append(word)
    
    if words_to_process:
        inputs = tokenizer(words_to_process, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        new_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        for word, embedding in zip(words_to_process, new_embeddings):
            embedding_cache[word] = embedding
            embeddings.append(embedding)
    
    return np.array(embeddings)

# 불용어 리스트
stopwords = ['스타', '벅스', '스타벅스', '스벅', '매장', '카페']

# 불용어 제거 함수
def remove_stopwords(nouns, stopwords):
    return [noun for noun in nouns if noun not in stopwords]

# 사용자 입력 명사 추출 함수 정의 및 불용어 제거 적용
def extract_nouns(user_input):
    okt = Okt()
    nouns = okt.nouns(user_input)
    filtered_nouns = remove_stopwords(nouns, stopwords)
    return filtered_nouns

# 빈 데이터 필터링 함수
def filter_data(data, user_input):
    filtered_data = data.copy()
    # 매장 타입 관련 필터링
    if '리저브' in user_input or 'Reserve' in user_input or 'reserve' in user_input:
        filtered_data = filtered_data[filtered_data['storeType'] == '리저브']
    if '일반' in user_input or 'Standard' in user_input or 'standard' in user_input:
        filtered_data = filtered_data[filtered_data['storeType'] == '일반']
    if '드라이브 스루' in user_input or '드라이브스루' in user_input or 'drivethrough' in user_input:
        filtered_data = filtered_data[filtered_data['storeType'] == '드라이브스루']

    # 위치 관련 필터링
    if '서울시' in user_input or '서울특별시' in user_input or '수도' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('서울특별시')] 
    if '부산' in user_input or '부산시' in user_input or '부산광역시' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('부산')]
    if '대구' in user_input or '대구시' in user_input or '대구광역시' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('대구')]
    if '인천' in user_input or '인천광역시' in user_input or '인천시' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('인천')]
    if '광주' in user_input or '광주광역시' in user_input or 'gwangju' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('광주광역시')]
    if '대전' in user_input or '대전시' in user_input or '대전광역시' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('대전')]
    if '울산' in user_input or '울산시' in user_input or '울산광역시' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('울산')]
    if '세종' in user_input or '세종특별시' in user_input or 'sejong' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('세종특별자치시')]
    if '경기' in user_input or '경기도' in user_input or '수도권' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('경기')]
    if '강원' in user_input or '강원도' in user_input or 'gangwon' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('강원')]
    if '충북' in user_input or '충청북도' in user_input or 'chungbuk' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('충청북도')]
    if '충남' in user_input or '충청남도' in user_input or 'chungnam' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('충청남도')]
    if '전북' in user_input or '전라북도' in user_input or 'jeonbuk' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('전라북도')]
    if '전남' in user_input or '전라남도' in user_input or 'jeonnam' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('전라남도')]
    if '경북' in user_input or '경상북도' in user_input or 'gyeongbuk' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('경상북도')]
    if '경남' in user_input or '경상남도' in user_input or 'gyeongnam' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('경상남도')]
    if '제주' in user_input or 'Jeju' in user_input or 'jeju' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('제주')]
    
    # 대한민국 시 목록으로 필터링
    
    # 경기도
    if '수원시' in user_input or '수원' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('수원시')]
    if '용인시' in user_input or '용인' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('용인시')]
    if '고양시' in user_input or '고양' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('고양시')]
    if '화성시' in user_input or '화성' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('화성시')]
    if '성남시' in user_input or '성남' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('성남시')]
    if '부천시' in user_input or '부천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('부천시')]
    if '남양주시' in user_input or '남양주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('남양주시')]
    if '안산시' in user_input or '안산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('안산시')]
    if '평택시' in user_input or '평택' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('평택시')]
    if '안양시' in user_input or '안양' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('안양시')]
    if '시흥시' in user_input or '시흥' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('시흥시')]
    if '파주시' in user_input or '파주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('파주시')]
    if '김포시' in user_input or '김포' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('김포시')]
    if '의정부시' in user_input or '의정부' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('의정부시')]
    if '광주시' in user_input or '경기 광주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('경기도 광주')]
    if '하남시' in user_input or '하남' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('하남시')]
    if '광명시' in user_input or '광명' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('광명시')]
    if '군포시' in user_input or '군포' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('군포시')]
    if '양주시' in user_input or '양주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('양주시')]
    if '오산시' in user_input or '오산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('오산시')]
    if '이천시' in user_input or '이천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('이천시')]
    if '안성시' in user_input or '안성' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('안성시')]
    if '구리시' in user_input or '구리' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('구리시')]
    if '의왕시' in user_input or '의왕' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('의왕시')]
    if '포천시' in user_input or '포천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('포천시')]

    
    # 강원특별자치도
    if '춘천시' in user_input or '춘천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('춘천시')]
    if '원주시' in user_input or '원주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('원주시')]
    if '강릉시' in user_input or '강릉' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('강릉시')]
    if '동해시' in user_input or '동해' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('동해시')]
    if '속초시' in user_input or '속초' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('속초시')]
    if '삼척시' in user_input or '삼척' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('삼척시')]
    
    # 전라남도
    if '목포시' in user_input or '목포' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('목포시')]
    if '여수시' in user_input or '여수' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('여수시')]
    if '순천시' in user_input or '순천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('순천시')]
    if '나주시' in user_input or '나주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('나주시')]
    if '광양시' in user_input or '광양' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('광양시')]
    # 전라북도
    if '전주시' in user_input or '전주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('전주시')]
    if '군산시' in user_input or '군산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('군산시')]
    if '익산시' in user_input or '익산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('익산시')]
    if '정읍시' in user_input or '정읍' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('정읍시')]
    if '남원시' in user_input or '남원' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('남원시')]
    if '김제시' in user_input or '김제' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('김제시')]

    # 경상북도
    if '포항시' in user_input or '포항' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('포항시')]
    if '경주시' in user_input or '경주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('경주시')]
    if '김천시' in user_input or '김천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('김천시')]
    if '안동시' in user_input or '안동' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('안동시')]
    if '구미시' in user_input or '구미' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('구미시')]
    if '영주시' in user_input or '영주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('영주시')]
    if '영천시' in user_input or '영천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('영천시')]
    if '상주시' in user_input or '상주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('상주시')]
    if '문경시' in user_input or '문경' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('문경시')]
    if '경산시' in user_input or '경산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('경산시')]

    # 경상남도
    if '창원시' in user_input or '창원' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('창원시')]
    if '진주시' in user_input or '진주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('진주시')]
    if '통영시' in user_input or '통영' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('통영시')]
    if '사천시' in user_input or '사천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('사천시')]
    if '김해시' in user_input or '김해' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('김해시')]
    if '밀양시' in user_input or '밀양' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('밀양시')]
    if '거제시' in user_input or '거제' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('거제시')]
    if '양산시' in user_input or '양산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('양산시')]

    # 충청남도 
    if '천안시' in user_input or '천안' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('천안시')]
    if '공주시' in user_input or '공주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('공주시')]
    if '보령시' in user_input or '보령' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('보령시')]
    if '아산시' in user_input or '아산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('아산시')]
    if '서산시' in user_input or '서산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('서산시')]
    if '논산시' in user_input or '논산' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('논산시')]
    if '계룡시' in user_input or '계룡' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('계룡시')]
    if '당진시' in user_input or '당진' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('당진시')]
    
    # 충청북도
    if '청주시' in user_input or '청주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('청주시')]
    if '충주시' in user_input or '충주' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('충주시')]
    if '제천시' in user_input or '제천' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('제천시')]

    
    # 서울시 지하철 역 입력시 구로 반환
    # 강남구
    if '삼성역' in user_input or '선릉역' in user_input or '역삼역' in user_input or '강남역' in user_input or '압구정역' in user_input or '신사역' in user_input or '매봉역' in user_input or '도곡역' in user_input or '대치역' in user_input or '학여울역' in user_input or '대청역' in user_input or '일원역' in user_input or '수서역' in user_input or '강남구청역' in user_input or '학동역' in user_input or '논현역' in user_input or '신논현역' in user_input or '언주역' in user_input or '선정릉역' in user_input or '삼성중앙역' in user_input or '봉은사역' in user_input or '압구정로데오역' in user_input or '한티역' in user_input or '구릉역' in user_input or '개포동역' in user_input or '대모산역' in user_input or '청담역' in user_input or '강남구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('강남구')]

    # 강동구
    if '천호역' in user_input or '강동역' in user_input or '길동역' in user_input or '굽은다리역' in user_input or '명일역' in user_input or '고덕역' in user_input or '상일동역' in user_input or '강일역' in user_input or '둔촌동역' in user_input or '암사역' in user_input or '강동구청역' in user_input or '둔촌오륜역' in user_input or '중앙보훈병원역' in user_input or '강동구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('강동구')]

    # 강북구
    if '미아사거리역' in user_input or '미아역' in user_input or '수유역' in user_input or '솔샘역' in user_input or '삼양사거리역' in user_input or '삼양역' in user_input or '화계역' in user_input or '가오리역' in user_input or '4.19민주묘지역' in user_input or '솔밭공원역' in user_input or '북한산우이역' in user_input or '강북구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('강북구')]

    # 강서구
    if '까치산역' in user_input or '방화역' in user_input or '개화산역' in user_input or '김포공항역' in user_input or '송정역' in user_input or '마곡역' in user_input or '발산역' in user_input or '우장산역' in user_input or '화곡역' in user_input or '공항시장역' in user_input or '신방화역' in user_input or '마곡나루역' in user_input or '양천향교역' in user_input or '가양역' in user_input or '증미역' in user_input or '등촌역' in user_input or '염창역' in user_input or '강서구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('강서구')]

    # 관악구
    if '낙성대역' in user_input or '서울대입구역' in user_input or '봉천역' in user_input or '신림역' in user_input or '당곡역' in user_input or '서원역' in user_input or '서울대벤처타운역' in user_input or '관악산역' in user_input or '관악구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('관악구')]

    # 광진구
    if '건대입구역' in user_input or '구의역' in user_input or '강변역' in user_input or '군자역' in user_input or '아차산역' in user_input or '광나루역' in user_input or '중곡역' in user_input or '어린이대공원역' in user_input or '뚝섬유원지역' in user_input or '광진구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('광진구')]
    # 구로구
    if '구로역' in user_input or '구일역' in user_input or '개봉역' in user_input or '오류동역' in user_input or '온수역' in user_input or '신도림역' in user_input or '구로디지털단지역' in user_input or '대림역' in user_input or '도림천역' in user_input or '남구로역' in user_input or '천왕역' in user_input or '구로구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('구로구')]

    # 금천구
    if '금천구청역' in user_input or '독산역' in user_input or '가산디지털단지역' in user_input or '금천구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('금천구')]
    
    # 노원구
    if '석계역' in user_input or '광운대역' in user_input or '월계역' in user_input or '노원역' in user_input or '상계역' in user_input or '당고개역' in user_input or '화랑대역' in user_input or '태릉입구역' in user_input or '수락산역' in user_input or '마들역' in user_input or '중계역' in user_input or '하계역' in user_input or '공릉역' in user_input or '노원구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('노원구')]

    # 도봉구
    if '녹천역' in user_input or '창동역' in user_input or '방학역' in user_input or '도봉역' in user_input or '도봉산역' in user_input or '쌍문역' in user_input or '도봉구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('도봉구')]

    # 동대문구
    if '신설동역' in user_input or '제기동역' in user_input or '청량리역' in user_input or '회기역' in user_input or '외대앞역' in user_input or '신이문역' in user_input or '용두역' in user_input or '답십리역' in user_input or '장한평역' in user_input or '동대문구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('동대문구')]
    
    # 동작구
    if '노량진역' in user_input or '사당역' in user_input or '신대방역' in user_input or '이수역' in user_input or '총신대입구역' in user_input or '동작역' in user_input or '남성역' in user_input or '숭실대입구역' in user_input or '상도역' in user_input or '장승배기역' in user_input or '신대방삼거리역' in user_input or '노들역' in user_input or '흑석역' in user_input or '보라매공원역' in user_input or '보라매병원역' in user_input or '동작구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('동작구')]

    # 마포구
    if '합정역' in user_input or '홍대입구역' in user_input or '신촌역' in user_input or '이대역' in user_input or '아현역' in user_input or '마포역' in user_input or '공덕역' in user_input or '애오개역' in user_input or '대흥역' in user_input or '광흥창역' in user_input or '상수역' in user_input or '망원역' in user_input or '마포구청역' in user_input or '월드컵경기장역' in user_input or '디지털미디어시티역' in user_input or '서강대역' in user_input or '마포구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('마포구')]

    # 서대문구
    if '충정로역' in user_input or '홍제역' in user_input or '무악재역' in user_input or '서대문역' in user_input or '가좌역' in user_input or '서대문구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('서대문구')]


    # 서초구
    if '교대역' in user_input or '서초역' in user_input or '방배역' in user_input or '잠원역' in user_input or '고속터미널역' in user_input or '남부터미널역' in user_input or '양재역' in user_input or '남태령역' in user_input or '반포역' in user_input or '내방역' in user_input or '구반포역' in user_input or '신반포역' in user_input or '사평역' in user_input or '양재시민의숲역' in user_input or '청계산입구역' in user_input or '서초구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('서초구')]


    # 성동구
    if '상왕십리역' in user_input or '왕십리역' in user_input or '한양대역' in user_input or '뚝섬역' in user_input or '성수역' in user_input or '용답역' in user_input or '신답역' in user_input or '금호역' in user_input or '옥수역' in user_input or '신금호역' in user_input or '행당역' in user_input or '마장역' in user_input or '응봉역' in user_input or '서울숲역' in user_input or '성동구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('성동구')]


    # 성북구
    if '한성대입구역' in user_input or '성신여대입구역' in user_input or '길음역' in user_input or '돌곶이역' in user_input or '상월곡역' in user_input or '월곡역' in user_input or '고려대역' in user_input or '안암역' in user_input or '보문역' in user_input or '북한산보국문역' in user_input or '정릉역' in user_input or '성북구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('성북구')]


    # 송파구
    if '잠실나루역' in user_input or '잠실역' in user_input or '잠실새내역' in user_input or '종합운동장역' in user_input or '가락시장역' in user_input or '경찰병원역' in user_input or '오금역' in user_input or '올림픽공원역' in user_input or '방이역' in user_input or '개롱역' in user_input or '거여역' in user_input or '마천역' in user_input or '몽촌토성역' in user_input or '석촌역' in user_input or '송파역' in user_input or '문정역' in user_input or '장지역' in user_input or '복정역' in user_input or '삼전역' in user_input or '석촌고분역' in user_input or '송파나루역' in user_input or '한성백제역' in user_input or '송파구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('송파구')]

    # 양천구
    if '양천구청역' in user_input or '신정네거리역' in user_input or '신정역' in user_input or '목동역' in user_input or '오목교역' in user_input or '신목동역' in user_input or '양천구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('양천구')]

    # 영등포구
    if '영등포역' in user_input or '신길역' in user_input or '대방역' in user_input or '문래역' in user_input or '영등포구청역' in user_input or '당산역' in user_input or '양평역' in user_input or '영등포시장역' in user_input or '여의도역' in user_input or '여의나루역' in user_input or '보라매역' in user_input or '신풍역' in user_input or '선유도역' in user_input or '국회의사당역' in user_input or '샛강역' in user_input or '서울지방병무청역' in user_input or '영등포구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('영등포구')]


    # 용산구
    if '용산역' in user_input or '남영역' in user_input or '서울역' in user_input or '이촌역' in user_input or '신용산역' in user_input or '삼각지역' in user_input or '숙대입구역' in user_input or '한강진역' in user_input or '이태원f역' in user_input or '녹사평역' in user_input or '효창공원앞역' in user_input or '서빙고역' in user_input or '한남역' in user_input or '용산구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('용산구')]

    # 은평구
    if '구파발역' in user_input or '연신내역' in user_input or '불광역' in user_input or '녹번역' in user_input or '디지털미디어시티역' in user_input or '증산역' in user_input or '새절역' in user_input or '응암역' in user_input or '구산역' in user_input or '독바위역' in user_input or '역촌역' in user_input or '응암역' in user_input or '수색역' in user_input or '은평구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('은평구')]


    # 종로구
    if '종각역' in user_input or '종로3가역' in user_input or '종로5가역' in user_input or '동대문역' in user_input or '동묘앞역' in user_input or '독립문역' in user_input or '경복궁역' in user_input or '안국역' in user_input or '혜화역' in user_input or '광화문역' in user_input or '창신역' in user_input or '종로구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('종로구')]


    # 중구
    if '서울역' in user_input or '시청역' in user_input or '을지로입구역' in user_input or '을지로3가역' in user_input or '을지로4가역' in user_input or '동대문역사문화공원역' in user_input or '신당역' in user_input or '충무로역' in user_input or '동대입구역' in user_input or '약수역' in user_input or '회현역' in user_input or '명동역' in user_input or '청구역' in user_input or '버티고개역' in user_input or '중구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('중구')]

    # 중랑구
    if '신내역' in user_input or '봉화산역' in user_input or '먹골역' in user_input or '중화역' in user_input or '상봉역' in user_input or '면목역' in user_input or '사가정역' in user_input or '용마산역' in user_input or '중랑역' in user_input or '망우역' in user_input or '양원역' in user_input or '중랑구' in user_input:
        filtered_data = filtered_data[filtered_data['storeAddress'].str.contains('중랑구')]



    # 시설 관련 필터링
    if '주차공간 있는' in user_input or '주차가능' in user_input or '주차' in user_input or '주차 가능' in user_input:
        filtered_data = filtered_data[filtered_data['parking'] == True]
    if '주차공간이없는' in user_input or '주차가불가' in user_input or '차대는곳 없는' in user_input or '주차 불가능' in user_input:
        filtered_data = filtered_data[filtered_data['parking'] == False]
    if '주차불가능' in user_input or '주차가 불가' in user_input or '주차못' in user_input or '주차 못' in user_input:
        filtered_data = filtered_data[filtered_data['parking'] == False]

    if '블론드' in user_input or '블론드 라떼' in user_input or '블론드 에스프레소' in user_input or '블론드 원두' in user_input:
        filtered_data = filtered_data[filtered_data['blonde'] == True]
    if '블론드 아닌' in user_input or '블론드 제외' in user_input or '블론드 제외한' in user_input or '블론드 없는' in user_input:
        filtered_data = filtered_data[filtered_data['blonde'] == False]

    if '피지오' in user_input or '피지오 드링크' in user_input or '피지오 음료' in user_input or '피지오 커피' in user_input:
        filtered_data = filtered_data[filtered_data['physio'] == True]
    if '피지오 아닌' in user_input or '피지오 제외' in user_input or '피지오 제외한' in user_input or '피지오 없는' in user_input:
        filtered_data = filtered_data[filtered_data['physio'] == False]

    if '콜드브루' in user_input or '콜드 브루' in user_input or '콜드브루 커피' in user_input or '콜드 브루 커피' in user_input:
        filtered_data = filtered_data[filtered_data['coldbrew'] == True]
    if '콜드브루 아닌' in user_input or '콜드브루 제외' in user_input or '콜드브루 제외한' in user_input or '콜드브루 없는' in user_input:
        filtered_data = filtered_data[filtered_data['coldbrew'] == False]

    if '현금불가' in user_input or '현금 불가' in user_input or '현금 사용 불가' in user_input or '현금 받지 않는' in user_input:
        filtered_data = filtered_data[filtered_data['noCash'] == True]
    if '현금가능' in user_input or '현금 가능' in user_input or '현금 사용 가능' in user_input or '현금 받는' in user_input:
        filtered_data = filtered_data[filtered_data['noCash'] == False]

    if '외화결제' in user_input or '외화 결제' in user_input or '외국 화폐 결제' in user_input or '외화 사용 가능' in user_input:
        filtered_data = filtered_data[filtered_data['foreignCash'] == True]
    if '외화결제 불가' in user_input or '외화 결제 불가' in user_input or '외국 화폐 사용 불가' in user_input or '외화 사용 불가능' in user_input:
        filtered_data = filtered_data[filtered_data['foreignCash'] == False]

    if '딜리버스' in user_input or '딜리버리' in user_input or '배달 가능한' in user_input or '배달 되는' in user_input:
        filtered_data = filtered_data[filtered_data['deliBus'] == True]
    if '딜리버스 아닌' in user_input or '딜리버리 불가' in user_input or '배달 불가' in user_input or '배달 안되는' in user_input:
        filtered_data = filtered_data[filtered_data['deliBus'] == False]

    if '친환경'in user_input or 'in코'in user_input or '환경 친화적'in user_input or '환경 보호'in user_input:
        filtered_data = filtered_data[filtered_data['eco'] == True]
    if '친환경 아닌'in user_input or 'in코 아닌'in user_input or '환경 친화적 아닌'in user_input or '환경 보호 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['eco'] == False]

    if '오후9시이후영업'in user_input or '야간영업'in user_input or '밤에 여는'in user_input or '밤늦게 여는'in user_input:
        filtered_data = filtered_data[filtered_data['close21'] == True]
    if '오후9시이후영업 아닌'in user_input or '야간영업 불가'in user_input or '밤에 닫는'in user_input or '일찍 닫는'in user_input:
        filtered_data = filtered_data[filtered_data['close21'] == False]

    if '펫존'in user_input or '반려동물 존'in user_input or '애완동물 존'in user_input or '반려동물 공간'in user_input:
        filtered_data = filtered_data[filtered_data['petZone'] == True]
    if '펫존 아닌'in user_input or '반려동물 존 아닌'in user_input or '애완동물 존 아닌'in user_input or '반려동물 공간 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['petZone'] == False]

    if '공항'in user_input or '공항 근처'in user_input or '공항 근방'in user_input or '공항 주변'in user_input:
        filtered_data = filtered_data[filtered_data['airport'] == True]
    if '공항 아닌'in user_input or '공항 근처 아닌'in user_input or '공항 근방 아닌'in user_input or '공항 주변 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['airport'] == False]

    if '해변가'in user_input or '바닷가'in user_input or '바다 근처'in user_input or '해안가'in user_input:
        filtered_data = filtered_data[filtered_data['seaside'] == True]
    if '해변가 아닌'in user_input or '바닷가 아닌'in user_input or '바다 근처 아닌'in user_input or '해안가 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['seaside'] == False]

    if '대학교'in user_input or '대학'in user_input or '대학 근처'in user_input or '학교 근처'in user_input:
        filtered_data = filtered_data[filtered_data['university'] == True]
    if '대학교 아닌'in user_input or '대학 아닌'in user_input or '대학 근처 아닌'in user_input or '학교 근처 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['university'] == False]

    if '터미널'in user_input or '버스터미널'in user_input or '터미널 근처'in user_input or '터미널 주변'in user_input:
        filtered_data = filtered_data[filtered_data['terminal'] == True]
    if '터미널 아닌'in user_input or '버스터미널 아닌'in user_input or '터미널 근처 아닌'in user_input or '터미널 주변 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['terminal'] == False]

    if '리조트'in user_input or '리조트 근처'in user_input or '리조트 주변'in user_input or '휴양지'in user_input:
        filtered_data = filtered_data[filtered_data['resort'] == True]
    if '리조트 아닌'in user_input or '리조트 근처 아닌'in user_input or '리조트 주변 아닌'in user_input or '휴양지 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['resort'] == False]

    if '병원'in user_input or '병원 근처'in user_input or '의료기관'in user_input or '의료시설'in user_input:
        filtered_data = filtered_data[filtered_data['hospital'] == True]
    if '병원 아닌'in user_input or '병원 근처 아닌'in user_input or '의료기관 아닌'in user_input or '의료시설 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['hospital'] == False]

    if '매장내'in user_input or '매장 내'in user_input or '가게 안'in user_input or '상점 내'in user_input:
        filtered_data = filtered_data[filtered_data['inStore'] == True]
    if '매장내 아닌'in user_input or '매장 내 아닌'in user_input or '가게 안 아닌'in user_input or '상점 내 아닌'in user_input:
        filtered_data = filtered_data[filtered_data['inStore'] == False]

    if '지하철' in user_input or '지하철역' in user_input or '지하철 근처' in user_input or '지하철 주변' in user_input:
        filtered_data = filtered_data[filtered_data['subway'] == True]
    if '지하철 아닌' in user_input or '지하철역 아닌' in user_input or '지하철 근처 아닌' in user_input or '지하철 주변 아닌' in user_input:
        filtered_data = filtered_data[filtered_data['subway'] == False]

    if '장애인편의시설' in user_input or '장애인 편의 시설' in user_input or '장애인 접근 가능' in user_input or '장애인 지원' in user_input:
        filtered_data = filtered_data[filtered_data['theDisabled'] == True]
    if '장애인편의시설 아닌' in user_input or '장애인 편의 시설 아닌' in user_input or '장애인 접근 불가' in user_input or '장애인 지원 불가' in user_input:
        filtered_data = filtered_data[filtered_data['theDisabled'] == False]

    if '공기청정기' in user_input or 'in어 클리너' in user_input or '공기 청정' in user_input or '공기 정화' in user_input:
        filtered_data = filtered_data[filtered_data['airCleaner'] == True]
    if '공기청정기 없는' in user_input or 'in어 클리너 없는' in user_input or '공기 청정 안 되는' in user_input or '공기 정화 안 되는' in user_input:
        filtered_data = filtered_data[filtered_data['airCleaner'] == False]

    if '전기차충전소' in user_input or '전기차 충전' in user_input or 'EV 충전' in user_input or '전기차 충전 가능' in user_input:
        filtered_data = filtered_data[filtered_data['electricVehicleCharging'] == True]
    if '전기차충전소 없는' in user_input or '전기차 충전 안 되는' in user_input or 'EV 충전 불가' in user_input or '전기차 충전 불가능' in user_input:
        filtered_data = filtered_data[filtered_data['electricVehicleCharging'] == False]

    return filtered_data

# 입력 해시를 생성하는 함수
def generate_input_hash(user_input):
    return hashlib.md5(user_input.encode()).hexdigest()

# 캐시된 결과를 저장하는 함수
def save_to_cache(input_hash, results):
    cache_file_path = os.path.join(CACHE_DIR, f"{input_hash}.json")
    with open(cache_file_path, 'w') as cache_file:
        json.dump(results, cache_file)

# 캐시된 결과를 불러오는 함수
def load_from_cache(input_hash):
    cache_file_path = os.path.join(CACHE_DIR, f"{input_hash}.json")
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as cache_file:
            return json.load(cache_file)
    return None

def recommend_stores(user_input):
    start_time = time.time()  # 시간 측정 시작

    # 사용자 입력 해시 생성
    input_hash = generate_input_hash(user_input)
    
    # 캐시된 결과 불러오기 시도
    cached_results = load_from_cache(input_hash)
    if cached_results:
        return cached_results

    # 사용자 입력에서 명사 추출
    nouns = extract_nouns(user_input)
    if not nouns:
        raise ValueError("No valid nouns extracted from user input.")
    
    # 사용자 입력 명사 임베딩
    user_embeddings = get_embeddings_with_cache(nouns)
    
    # 원본 데이터 로드
    file_path = './data/스타벅스추천모델빈도.csv'  # 이 경로를 실제 데이터 파일 경로로 수정하세요.
    data = pd.read_csv(file_path)
    
    # 데이터 필터링
    filtered_data = filter_data(data, user_input)
    
    # 각 매장의 유사도 계산
    store_scores = []
    for index, row in filtered_data.iterrows():
        frequency_dict = ast.literal_eval(row['frequency'])
        store_score = 0
        
        for noun, freq in frequency_dict.items():
            if noun in embedding_cache:
                word_embedding = embedding_cache[noun]
            else:
                word_embedding = get_embeddings_with_cache([noun])[0]
            
            similarities = cosine_similarity(user_embeddings, [word_embedding])
            max_similarity = similarities.max()
            
            if max_similarity >= 0.98:  # 유사도 기준치
                store_score += freq
        
        store_scores.append(store_score)
        
        # 매장 유사도 계산 중간 결과 출력
        print(f"매장 {index} - 점수: {store_score}")
    
    # 각 매장의 점수를 데이터프레임에 추가
    filtered_data['score'] = store_scores
    
    # 추천 매장 정렬
    recommended_stores = filtered_data.sort_values(by='score', ascending=False)
    
    end_time = time.time()  # 시간 측정 종료
    print(f"추천 계산에 소요된 시간: {end_time - start_time:.2f}초")

    # 추천 결과 저장
    results = recommended_stores[['Store_Name', 'score']].head(10).to_dict(orient='records')
    save_to_cache(input_hash, results)

    # 추천 결과 반환
    return results

if __name__ == "__main__":
    user_input = input("사용자 입력을 입력하세요: ")
    recommendations = recommend_stores(user_input)
    print("추천 매장:", recommendations)