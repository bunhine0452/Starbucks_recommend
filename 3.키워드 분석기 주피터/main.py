# main.py

import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from io import StringIO

# Jupyter Notebook에서 이벤트 루프를 여러 번 실행할 수 있도록 설정
nest_asyncio.apply()

# FastAPI 애플리케이션 초기화
app = FastAPI()

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 형태소 분석기 초기화
okt = Okt()

# 형태소 분석 및 명사 추출 함수
def extract_nouns(text):
    tokens = okt.pos(text)
    nouns = [word for word, pos in tokens if pos in ['Noun']]
    return ' '.join(nouns)

# 불용어 목록 생성 함수
def generate_stopwords(nouns):
    noun_counts = Counter(nouns.split())
    total_nouns = len(noun_counts)
    top_1_percent = int(total_nouns * 0.01)
    bottom_1_percent = int(total_nouns * 0.01)
    
    stopwords = [noun for noun, count in noun_counts.most_common(top_1_percent)]
    stopwords += [noun for noun, count in noun_counts.most_common()[:-bottom_1_percent-1:-1]]
    return stopwords

# 불용어 제거된 명사 추출 함수
def filter_nouns(nouns, stopwords):
    return ' '.join([noun for noun in nouns.split() if noun not in stopwords])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract-nouns/")
async def extract_nouns_endpoint(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    data = pd.read_csv(StringIO(content.decode('utf-8')))
    data['nouns'] = data['Content'].apply(extract_nouns)
    
    result_html = data.to_html()
    return templates.TemplateResponse("result.html", {"request": request, "result": result_html})

@app.post("/generate-stopwords/")
async def generate_stopwords_endpoint(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    data = pd.read_csv(StringIO(content.decode('utf-8')))
    
    all_nouns = ' '.join(data['nouns'])
    stopwords = generate_stopwords(all_nouns)
    data['filtered_nouns'] = data['nouns'].apply(lambda x: filter_nouns(x, stopwords))
    
    result_html = data.to_html()
    return templates.TemplateResponse("result.html", {"request": request, "result": result_html})

# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
