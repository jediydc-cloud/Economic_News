"""
경제 뉴스 브리프
GitHub Actions 실행용 스크립트
- 경제/증시/금리/부동산/기업 뉴스 자동 수집
- 중요 뉴스 선별
- AI 한국어 요약 및 투자 시사점 생성
- watchlist 후보 생성
- docs/data.json 저장
"""

import os
import re
import json
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import pandas as pd
from bs4 import BeautifulSoup
import feedparser
from langdetect import detect, DetectorFactory
from dateutil import parser as dateparser
from google import genai

# ──────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0
KST = ZoneInfo("Asia/Seoul")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
client = genai.Client(api_key=GEMINI_API_KEY)

TOP_NEWS_COUNT = 6
CANDIDATE_POOL_SIZE = 10
CANDIDATE_MIN_DOMESTIC = 3
CANDIDATE_MIN_GLOBAL = 3
FINAL_MIN_DOMESTIC = 2
FINAL_MIN_GLOBAL = 2
WATCHLIST_COUNT = 4
MAX_ARTICLES_PER_FEED = 12

GEMINI_RETRIES = 3
GEMINI_WAIT_SECONDS = 18

OUTPUT_PATH = "docs/data.json"

# ──────────────────────────────────────────
# RSS 피드 설정
# Google News 검색형 RSS 중심
# ──────────────────────────────────────────
def build_google_news_rss(query: str, hl: str, gl: str, ceid: str) -> str:
    return f"https://news.google.com/rss/search?q={quote(query)}&hl={hl}&gl={gl}&ceid={ceid}"

RSS_FEEDS = [
    {
        "source": "Google News Korea",
        "region": "국내",
        "default_lang": "ko",
        "feed_hint": "국내 거시경제",
        "url": build_google_news_rss(
            "한국 경제 물가 수출 경기 소비 생산 성장률",
            "ko", "KR", "KR:ko"
        ),
    },
    {
        "source": "Google News Korea",
        "region": "국내",
        "default_lang": "ko",
        "feed_hint": "증시·금리",
        "url": build_google_news_rss(
            "한국 증시 금리 환율 코스피 코스닥 한국은행",
            "ko", "KR", "KR:ko"
        ),
    },
    {
        "source": "Google News Korea",
        "region": "국내",
        "default_lang": "ko",
        "feed_hint": "부동산·정책",
        "url": build_google_news_rss(
            "한국 부동산 정책 대출 세제 재건축 공급",
            "ko", "KR", "KR:ko"
        ),
    },
    {
        "source": "Google News Korea",
        "region": "국내",
        "default_lang": "ko",
        "feed_hint": "기업·산업",
        "url": build_google_news_rss(
            "한국 기업 실적 투자 반도체 자동차 배터리 AI",
            "ko", "KR", "KR:ko"
        ),
    },
    {
        "source": "Google News Global",
        "region": "해외",
        "default_lang": "en",
        "feed_hint": "글로벌 경제",
        "url": build_google_news_rss(
            "global economy inflation recession IMF World Bank central bank",
            "en", "US", "US:en"
        ),
    },
    {
        "source": "Google News Global",
        "region": "해외",
        "default_lang": "en",
        "feed_hint": "증시·금리",
        "url": build_google_news_rss(
            "US markets treasury yields fed interest rates Nasdaq S&P 500",
            "en", "US", "US:en"
        ),
    },
    {
        "source": "Google News Global",
        "region": "해외",
        "default_lang": "en",
        "feed_hint": "기업·산업",
        "url": build_google_news_rss(
            "earnings semiconductor AI energy EV supply chain industry",
            "en", "US", "US:en"
        ),
    },
]

# ──────────────────────────────────────────
# 카테고리 / 가중치
# ──────────────────────────────────────────
CATEGORY_RULES = {
    "국내 거시경제": [
        "물가", "고용", "수출", "수입", "생산", "소비", "성장률", "경기", "내수", "무역",
        "inflation", "cpi", "ppi", "gdp", "employment", "trade", "exports", "imports"
    ],
    "글로벌 경제": [
        "세계경제", "글로벌", "중국경제", "미국경제", "유럽경제", "recession", "imf", "world bank",
        "global economy", "world economy", "central bank", "tariff", "geopolitical"
    ],
    "증시·금리": [
        "증시", "주식", "코스피", "코스닥", "금리", "환율", "달러", "채권", "fed", "fomc",
        "nasdaq", "s&p", "dow", "yield", "treasury", "rate cut", "rate hike", "stocks", "market"
    ],
    "부동산·정책": [
        "부동산", "주택", "아파트", "재건축", "재개발", "공급", "대출", "세제", "규제", "정책",
        "mortgage", "housing", "property", "real estate", "tax", "regulation", "policy"
    ],
    "기업·산업": [
        "실적", "투자", "반도체", "배터리", "자동차", "조선", "에너지", "ai", "인공지능", "공장",
        "earnings", "guidance", "semiconductor", "chip", "battery", "automaker", "factory",
        "cloud", "software", "energy", "industry"
    ],
}
VALID_CATEGORIES = set(CATEGORY_RULES.keys()) | {"기타"}

SOURCE_WEIGHT = {
    "Google News Korea": 1.8,
    "Google News Global": 1.8,
}

CATEGORY_WEIGHT = {
    "국내 거시경제": 2.3,
    "글로벌 경제": 2.2,
    "증시·금리": 2.5,
    "부동산·정책": 2.0,
    "기업·산업": 2.1,
    "기타": 1.0,
}

BAD_UI_PHRASES = [
    "[번역 대기]",
    "[번역 필요]",
    "자동 fallback",
    "분석 대기",
    "요약 생성 실패",
    "fallback",
]

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "under", "amid",
    "news", "said", "says", "will", "after", "about", "more", "than", "their", "its",
    "있다", "했다", "한다", "위해", "대한", "관련", "통해", "이번", "에서", "으로", "까지",
    "및", "등", "것", "수", "더", "한", "해", "고", "를", "이", "은", "는", "가", "도"
}

SECTOR_STOCK_MAP = {
    "반도체": ["삼성전자", "SK하이닉스"],
    "배터리": ["LG에너지솔루션", "삼성SDI", "포스코퓨처엠"],
    "자동차": ["현대차", "기아"],
    "은행": ["KB금융", "신한지주", "하나금융지주"],
    "증권": ["미래에셋증권", "한국금융지주", "NH투자증권"],
    "건설": ["현대건설", "GS건설", "대우건설"],
    "부동산": ["현대건설", "DL이앤씨", "GS건설"],
    "에너지": ["SK이노베이션", "S-Oil", "한국전력"],
    "AI": ["네이버", "카카오", "삼성전자"],
    "클라우드": ["네이버", "카카오", "더존비즈온"],
    "방산": ["한화에어로스페이스", "LIG넥스원", "현대로템"],
    "조선": ["HD한국조선해양", "한화오션", "삼성중공업"],
}

# ──────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────
def clean_text(text):
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title_key(title):
    title = clean_text(title).lower()
    title = re.sub(r"\[[^\]]+\]", " ", title)
    title = re.sub(r"\([^)]+\)", " ", title)
    title = re.sub(r"\s*[-|]\s*[^-|]+$", "", title)
    title = re.sub(r"[^0-9a-zA-Z가-힣 ]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def make_issue_dedup_key(title):
    base = normalize_title_key(title)
    tokens = [t for t in base.split() if len(t) >= 2]
    return " ".join(tokens[:7])


def parse_pubdate(date_text):
    try:
        dt = dateparser.parse(str(date_text))
        if dt is None:
            return pd.NaT
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.astimezone(KST).replace(tzinfo=None)
    except Exception:
        return pd.NaT


def format_kst(dt):
    if pd.isna(dt):
        return ""
    return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M")


def detect_language_safe(text, default="ko"):
    text = clean_text(text)
    if not text or len(text) < 5:
        return default
    try:
        lang = detect(text)
    except Exception:
        lang = default

    if lang.startswith("ko"):
        return "ko"
    if lang.startswith("en"):
        return "en"
    return default


def guess_category(text):
    text_clean = clean_text(text).lower()
    scores = {cat: 0 for cat in CATEGORY_RULES}

    for category, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            kw = kw.lower().strip()
            if re.search(r"[a-z]", kw):
                if re.search(rf"\b{re.escape(kw)}\b", text_clean):
                    scores[category] += 1
            else:
                if kw in text_clean:
                    scores[category] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "기타"


def recency_score(dt):
    if pd.isna(dt):
        return 0.5
    now = datetime.now(KST).replace(tzinfo=None)
    hours = (now - pd.Timestamp(dt).to_pydatetime()).total_seconds() / 3600
    hours = max(hours, 0)
    if hours <= 6:
        return 5.0
    if hours <= 12:
        return 4.2
    if hours <= 24:
        return 3.5
    if hours <= 48:
        return 2.4
    if hours <= 72:
        return 1.2
    return 0.5


def description_score(desc):
    length = len(clean_text(desc))
    if length >= 180:
        return 2.5
    if length >= 120:
        return 2.0
    if length >= 60:
        return 1.2
    if length >= 20:
        return 0.7
    return 0.2


def duplicate_score(cluster_count):
    if pd.isna(cluster_count):
        return 0.0
    return min((int(cluster_count) - 1) * 1.0, 3.5)


def score_news_row(row):
    score = 0.0
    score += recency_score(row["발행일"])
    score += description_score(row["기사설명"])
    score += SOURCE_WEIGHT.get(row["출처"], 1.0)
    score += CATEGORY_WEIGHT.get(row["카테고리_규칙"], 1.0)
    score += duplicate_score(row["제목클러스터건수"])
    return round(score, 2)


def clean_user_text(text, fallback=""):
    text = clean_text(text)
    if not text:
        return fallback
    lowered = text.lower()
    if any(bad.lower() in lowered for bad in BAD_UI_PHRASES):
        return fallback
    return text


def safe_int(value, default=0, minimum=None, maximum=None):
    try:
        value = int(value)
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def normalize_category(value, fallback="기타"):
    value = clean_text(value)
    if value in VALID_CATEGORIES:
        return value
    return fallback if fallback in VALID_CATEGORIES else "기타"


def shorten_text(text, max_len=150):
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def split_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?。！？])\s+|(?<=다\.)\s+", text)
    return [p.strip() for p in parts if p.strip()]


def fallback_keywords(title, desc, top_n=4):
    text = f"{clean_text(title)} {clean_text(desc)}".lower()
    tokens = re.findall(r"[0-9a-zA-Z가-힣]{2,}", text)
    freq = {}

    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) < 2:
            continue
        freq[token] = freq.get(token, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]


def infer_market_impact(title, desc):
    text = f"{clean_text(title)} {clean_text(desc)}".lower()

    negative_keywords = [
        "inflation", "관세", "전쟁", "분쟁", "지정학", "리스크", "급락", "하락", "긴축",
        "rate hike", "downgrade", "slowdown", "recession", "tariff", "crisis", "selloff"
    ]
    positive_keywords = [
        "rate cut", "인하", "부양", "상승", "호조", "실적개선", "투자확대", "수주", "지원",
        "stimulus", "upgrade", "growth", "surge", "beat", "expansion"
    ]

    if any(k in text for k in negative_keywords):
        return "부정"
    if any(k in text for k in positive_keywords):
        return "긍정"
    return "중립"


def infer_investment_view(market_impact):
    if market_impact == "긍정":
        return "관심"
    if market_impact == "부정":
        return "주의"
    return "중립"


def fallback_related_sector(category, title, desc):
    text = f"{clean_text(title)} {clean_text(desc)}".lower()

    if "반도체" in text or "semiconductor" in text or "chip" in text:
        return "반도체"
    if "배터리" in text or "battery" in text or "ev" in text:
        return "2차전지"
    if "자동차" in text or "automaker" in text or "car" in text:
        return "자동차"
    if "은행" in text or "금융" in text or "bank" in text:
        return "은행"
    if "증권" in text or "brokerage" in text:
        return "증권"
    if "부동산" in text or "주택" in text or "housing" in text or "property" in text:
        return "부동산/건설"
    if "에너지" in text or "oil" in text or "gas" in text:
        return "에너지"
    if "ai" in text or "인공지능" in text or "cloud" in text:
        return "AI/소프트웨어"

    if category == "증시·금리":
        return "금융/증권"
    if category == "부동산·정책":
        return "부동산/건설"
    if category == "기업·산업":
        return "제조/산업"
    return "시장 전반"


def fallback_related_stocks(sector, title, desc):
    sector = clean_text(sector)
    title_desc = f"{clean_text(title)} {clean_text(desc)}"

    for key, stocks in SECTOR_STOCK_MAP.items():
        if key in sector or key in title_desc:
            return stocks[:3]

    if "금융" in sector:
        return ["KB금융", "신한지주"]
    if "건설" in sector:
        return ["현대건설", "GS건설"]
    if "산업" in sector:
        return ["삼성전자", "현대차"]
    return []


def fallback_investment_reason(category, title, desc, impact, sector):
    if category == "증시·금리":
        return "금리와 유동성 변화는 시장 전반 밸류에이션과 자금 흐름에 직접 영향을 줄 수 있어 주목할 필요가 있습니다."
    if category == "국내 거시경제":
        return "국내 경기와 물가, 수출 흐름은 실적 민감 업종과 정책 방향에 연결되므로 투자 판단의 기초 자료가 됩니다."
    if category == "글로벌 경제":
        return "글로벌 경기와 지정학 변수는 국내 증시와 수출주에 연쇄적으로 영향을 줄 수 있습니다."
    if category == "부동산·정책":
        return "정책 변화는 건설·금융·소비 관련 업종의 수급과 실적 기대에 영향을 줄 수 있습니다."
    if category == "기업·산업":
        return f"{sector} 관련 업종과 종목에 직접적인 기대 또는 우려를 반영할 수 있는 뉴스입니다."
    return "시장 참여자들의 기대 심리와 업종별 수급 흐름에 참고할 만한 이슈입니다."


def fallback_key_signal(category, impact):
    if category == "증시·금리":
        return "금리 방향성과 시장 유동성 흐름 확인"
    if category == "국내 거시경제":
        return "국내 경기지표와 정책 대응 속도 점검"
    if category == "글로벌 경제":
        return "미국·중국 등 주요국 거시 변수와 지정학 리스크 확인"
    if category == "부동산·정책":
        return "정책 변화가 수요·대출·공급에 미치는 영향 점검"
    if category == "기업·산업":
        return "해당 업종 실적 모멘텀과 투자 확대 여부 확인"
    return "시장 심리 변화 여부 확인"


def fallback_summary_lines(title, desc, category):
    title = clean_text(title)
    desc = clean_text(desc)
    category = clean_text(category) or "기타"

    sentences = split_sentences(desc)

    if len(sentences) >= 3:
        lines = sentences[:3]
    elif len(sentences) == 2:
        lines = [sentences[0], sentences[1], f"{category} 관점에서 후속 흐름을 지켜볼 필요가 있습니다."]
    elif len(sentences) == 1:
        lines = [
            title if title else sentences[0],
            sentences[0],
            f"{category} 관련 투자 판단에 참고할 만한 흐름입니다.",
        ]
    else:
        lines = [
            title if title else "주요 경제 이슈입니다.",
            shorten_text(desc, 100) if desc else "핵심 내용 확인이 필요한 기사입니다.",
            f"{category} 관련 시장 반응을 지켜볼 필요가 있습니다.",
        ]

    cleaned = []
    for line in lines:
        line = clean_user_text(line, "")
        if not line:
            line = "관련 흐름을 추가로 확인할 필요가 있습니다."
        cleaned.append(shorten_text(line, 120))

    while len(cleaned) < 3:
        cleaned.append(f"{category} 관련 흐름을 추가 확인할 필요가 있습니다.")

    return cleaned[:3]


def normalize_keywords(value, title="", desc=""):
    if isinstance(value, list):
        keywords = [clean_text(v) for v in value if clean_text(v)]
    else:
        value = clean_text(value)
        keywords = [v.strip() for v in value.split(",") if v.strip()] if value else []

    if not keywords:
        keywords = fallback_keywords(title, desc, top_n=4)

    deduped = []
    for kw in keywords:
        if kw not in deduped:
            deduped.append(kw)

    return deduped[:5]


def build_project_overview():
    return {
        "project_name": "경제 뉴스 브리프",
        "purpose": "국내외 경제·증시·금리·부동산·기업 뉴스를 자동 수집·분석하여 투자 판단에 참고할 수 있는 일일 브리프를 제공하는 시스템",
        "main_features": [
            "경제 뉴스 자동 수집",
            "중요 뉴스 선별 및 한국어 요약",
            "투자 시사점 및 관련 업종/종목 정리",
            "관심 종목 watchlist 자동 제안"
        ],
        "final_deliverable": "Economic_News 홈페이지용 data.json"
    }

# ──────────────────────────────────────────
# Gemini 호출
# ──────────────────────────────────────────
def extract_json_from_text(raw):
    raw = str(raw).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```json\s*", "", raw).strip()
    raw = re.sub(r"^```\s*", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except Exception:
            pass

    arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group())
        except Exception:
            pass

    raise ValueError(f"JSON 파싱 실패 | 원문 앞 200자: {raw[:200]}")


def call_gemini_json(prompt, retries=GEMINI_RETRIES, wait=GEMINI_WAIT_SECONDS):
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)
            return extract_json_from_text(text)
        except Exception as e:
            last_err = e
            err_str = str(e)
            err_lower = err_str.lower()

            if "401" in err_str or "unauthenticated" in err_lower:
                raise RuntimeError(f"Gemini 인증 오류: {err_str}") from e

            if "404" in err_str and "not found" in err_lower:
                raise RuntimeError(f"Gemini 모델/리소스 오류: {err_str}") from e

            matched = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
            if matched:
                retry_wait = min(float(matched.group(1)) + 3, 90)
            elif "429" in err_str or "resource_exhausted" in err_lower:
                retry_wait = min(wait * attempt, 90)
            else:
                retry_wait = min(5 * attempt, 30)

            print(f"    ⚠️ Gemini 시도 {attempt}/{retries} 실패 → {retry_wait:.0f}초 대기")
            time.sleep(retry_wait)

    raise RuntimeError(f"Gemini 호출 최종 실패: {last_err}")

# ──────────────────────────────────────────
# 기사 배치 분석
# ──────────────────────────────────────────
def analyze_articles_batch(df_candidates):
    articles_text = []
    for i, (_, row) in enumerate(df_candidates.iterrows(), start=1):
        articles_text.append(
            f"[기사{i}]\n"
            f"출처: {row['출처']} / 국내외: {row['국내외구분']} / 언어: {row['언어']}\n"
            f"피드 힌트: {row['피드카테고리']}\n"
            f"발행일: {row['발행일_표준']}\n"
            f"제목(원문): {row['기사제목']}\n"
            f"설명(원문): {row['기사설명']}"
        )

    prompt = f"""
너는 한국 개인투자자를 위해 매일 아침 경제 뉴스 브리프를 정리하는 수석 애널리스트다.
아래 기사 {len(df_candidates)}개를 분석하여 반드시 JSON 배열로만 출력하라.
설명 문장, 코드블록, 마크다운 없이 JSON 배열만 출력하라.

[공통 출력 규칙]
1) 모든 출력은 한국어 중심으로 작성
2) translated_title_ko: 자연스러운 한국어 제목
3) translated_description_ko: 자연스러운 한국어 설명
4) summary_3lines: 정확히 3개 요소
5) category: 다음 중 하나만 사용
   ["국내 거시경제","글로벌 경제","증시·금리","부동산·정책","기업·산업","기타"]
6) importance_score_ai: 1~10 정수
7) market_impact: ["긍정","중립","부정"] 중 하나
8) related_sector: 업종/섹터 이름
9) related_stocks: 0~3개 종목명
10) investment_view: ["관심","중립","주의"] 중 하나
11) investment_reason: 왜 투자 관점에서 봐야 하는지 1문장
12) key_signal: 오늘 체크해야 할 핵심 포인트 1문장
13) keywords: 3~5개

[분석할 기사 목록]
{chr(10).join(articles_text)}

[JSON 배열 스키마]
[
  {{
    "idx": 1,
    "translated_title_ko": "한국어 제목",
    "translated_description_ko": "한국어 설명",
    "summary_3lines": ["핵심사실", "배경·맥락", "투자 시사점"],
    "category": "증시·금리",
    "importance_score_ai": 8,
    "market_impact": "중립",
    "related_sector": "반도체",
    "related_stocks": ["삼성전자", "SK하이닉스"],
    "investment_view": "관심",
    "investment_reason": "왜 투자 관점에서 중요한지",
    "key_signal": "오늘 체크할 포인트",
    "keywords": ["금리", "환율", "반도체"]
  }}
]
""".strip()

    try:
        data = call_gemini_json(prompt)
        if not isinstance(data, list):
            raise ValueError("배열 응답이 아님")
        print(f"  ✅ 배치 분석 성공: {len(data)}건 반환")
        return data
    except Exception as e:
        print(f"  ⚠️ 배치 분석 실패: {e}")
        return None


def build_article_analysis(row, item=None):
    title = clean_text(row.get("기사제목", ""))
    desc = clean_text(row.get("기사설명", ""))
    rule_category = normalize_category(row.get("카테고리_규칙", "기타"), "기타")
    language = row.get("언어", "ko")

    fallback_lines = fallback_summary_lines(title, desc, rule_category)
    fallback_impact = infer_market_impact(title, desc)
    fallback_view = infer_investment_view(fallback_impact)
    fallback_sector = fallback_related_sector(rule_category, title, desc)
    fallback_stocks = fallback_related_stocks(fallback_sector, title, desc)
    fallback_reason = fallback_investment_reason(rule_category, title, desc, fallback_impact, fallback_sector)
    fallback_signal = fallback_key_signal(rule_category, fallback_impact)
    fallback_kw = fallback_keywords(title, desc, top_n=4)

    if isinstance(item, dict):
        translated_title = clean_user_text(item.get("translated_title_ko", ""), title)
        translated_desc = clean_user_text(item.get("translated_description_ko", ""), desc)

        lines = item.get("summary_3lines", [])
        if not isinstance(lines, list):
            lines = fallback_lines
        lines = [clean_user_text(v, "") for v in lines if clean_user_text(v, "")]
        if len(lines) != 3:
            lines = fallback_lines

        category = normalize_category(item.get("category", ""), rule_category)
        ai_importance = safe_int(item.get("importance_score_ai", 5), default=5, minimum=1, maximum=10)

        market_impact = clean_user_text(item.get("market_impact", ""), fallback_impact)
        if market_impact not in ["긍정", "중립", "부정"]:
            market_impact = fallback_impact

        investment_view = clean_user_text(item.get("investment_view", ""), fallback_view)
        if investment_view not in ["관심", "중립", "주의"]:
            investment_view = fallback_view

        related_sector = clean_user_text(item.get("related_sector", ""), fallback_sector)
        related_stocks = item.get("related_stocks", [])
        if not isinstance(related_stocks, list):
            related_stocks = []
        related_stocks = [clean_text(v) for v in related_stocks if clean_text(v)]
        if not related_stocks:
            related_stocks = fallback_stocks

        investment_reason = clean_user_text(item.get("investment_reason", ""), fallback_reason)
        key_signal = clean_user_text(item.get("key_signal", ""), fallback_signal)
        keywords = normalize_keywords(item.get("keywords", []), title=title, desc=desc)

        return {
            "row_id": row["row_id"],
            "번역제목": translated_title if translated_title else title,
            "번역설명": translated_desc if translated_desc else desc,
            "요약1": lines[0],
            "요약2": lines[1],
            "요약3": lines[2],
            "3줄요약": "\n".join(lines),
            "카테고리": category,
            "AI중요도": ai_importance,
            "시장영향": market_impact,
            "관련업종": related_sector,
            "관련종목": related_stocks,
            "투자의견": investment_view,
            "투자근거": investment_reason,
            "핵심시그널": key_signal,
            "AI핵심키워드": keywords,
            "제목상태": "번역완료" if language == "en" and translated_title != title else ("원문한국어" if language == "ko" else "원문표시"),
            "요약상태": "AI요약",
            "분석상태": "ai"
        }

    return {
        "row_id": row["row_id"],
        "번역제목": title,
        "번역설명": desc,
        "요약1": fallback_lines[0],
        "요약2": fallback_lines[1],
        "요약3": fallback_lines[2],
        "3줄요약": "\n".join(fallback_lines),
        "카테고리": rule_category,
        "AI중요도": 5,
        "시장영향": fallback_impact,
        "관련업종": fallback_sector,
        "관련종목": fallback_stocks,
        "투자의견": fallback_view,
        "투자근거": fallback_reason,
        "핵심시그널": fallback_signal,
        "AI핵심키워드": fallback_kw,
        "제목상태": "원문한국어" if language == "ko" else "원문표시",
        "요약상태": "기본정리",
        "분석상태": "fallback"
    }

# ──────────────────────────────────────────
# 일일 브리프
# ──────────────────────────────────────────
def generate_daily_brief(df_top):
    records_text = []
    for _, row in df_top.iterrows():
        records_text.append(
            f"- [{row['국내외구분']}] {row['번역제목']} / 카테고리:{row['카테고리']} / 영향:{row['시장영향']} / 의견:{row['투자의견']} / 요약:{row['3줄요약'].replace(chr(10), ' | ')}"
        )

    prompt = f"""
너는 매일 아침 경제 뉴스 브리프를 작성하는 한국 주식시장 전문 애널리스트다.
아래 주요 뉴스를 바탕으로 반드시 JSON 객체만 출력하라.

[주요 뉴스]
{chr(10).join(records_text)}

[작성 규칙]
- 모두 한국어
- 투자자가 빠르게 읽고 방향을 잡을 수 있어야 함
- 과장 금지
- 짧지만 핵심적인 문장
- 오늘의한줄은 한 문장

[JSON 스키마]
{{
  "총평": "string",
  "국내거시경제": "string",
  "글로벌경제": "string",
  "증시금리": "string",
  "부동산정책": "string",
  "기업산업": "string",
  "오늘의투자시사점": "string",
  "오늘의한줄": "string"
}}
""".strip()

    fallback = {
        "총평": "오늘 시장은 거시 변수와 금리, 정책 변화, 기업 모멘텀이 동시에 반영되는 흐름으로 보입니다.",
        "국내거시경제": "국내 경기와 물가, 수출 흐름은 정책 기대와 함께 업종별 차별화를 만들 수 있습니다.",
        "글로벌경제": "미국 금리와 글로벌 경기, 지정학 변수는 국내 증시의 방향성에 계속 영향을 줄 가능성이 큽니다.",
        "증시금리": "금리와 환율 흐름은 성장주와 금융주, 수출주 밸류에이션에 직접적인 변수입니다.",
        "부동산정책": "부동산과 정책 변화는 건설, 금융, 소비 관련 업종에 파급될 수 있어 점검이 필요합니다.",
        "기업산업": "기업 실적과 산업 투자 뉴스는 개별 종목과 섹터별 모멘텀을 구분하는 기준이 됩니다.",
        "오늘의투자시사점": "단기 뉴스 반응보다 금리·실적·정책과 연결되는 업종별 흐름을 우선적으로 점검하는 것이 좋겠습니다.",
        "오늘의한줄": "오늘은 거시 변수와 업종 모멘텀을 함께 읽어야 하는 장입니다."
    }

    try:
        data = call_gemini_json(prompt)
        if not isinstance(data, dict):
            return fallback

        normalized = {
            "총평": clean_user_text(data.get("총평", ""), fallback["총평"]),
            "국내거시경제": clean_user_text(data.get("국내거시경제", ""), fallback["국내거시경제"]),
            "글로벌경제": clean_user_text(data.get("글로벌경제", ""), fallback["글로벌경제"]),
            "증시금리": clean_user_text(data.get("증시금리", ""), fallback["증시금리"]),
            "부동산정책": clean_user_text(data.get("부동산정책", ""), fallback["부동산정책"]),
            "기업산업": clean_user_text(data.get("기업산업", ""), fallback["기업산업"]),
            "오늘의투자시사점": clean_user_text(data.get("오늘의투자시사점", ""), fallback["오늘의투자시사점"]),
            "오늘의한줄": clean_user_text(data.get("오늘의한줄", ""), fallback["오늘의한줄"]),
        }
        return normalized
    except Exception:
        return fallback

# ──────────────────────────────────────────
# Watchlist 생성
# ──────────────────────────────────────────
def generate_watchlist(df_top, count=4):
    records_text = []
    for _, row in df_top.iterrows():
        records_text.append(
            f"- 제목:{row['번역제목']} / 카테고리:{row['카테고리']} / 시장영향:{row['시장영향']} / 관련업종:{row['관련업종']} / 관련종목:{', '.join(row['관련종목']) if isinstance(row['관련종목'], list) else ''} / 투자의견:{row['투자의견']} / 투자근거:{row['투자근거']}"
        )

    prompt = f"""
너는 한국 개인투자자를 위한 아침 리포트 애널리스트다.
아래 주요 뉴스로부터 오늘 관심 있게 볼 watchlist 후보 {count}개를 뽑아라.
반드시 JSON 객체만 출력하라.

[주요 뉴스]
{chr(10).join(records_text)}

[규칙]
- 모두 한국어
- opinion은 ["관심","중립","주의"] 중 하나
- 너무 공격적인 추천 표현 금지
- 가능하면 국내 상장 종목 우선
- 뉴스와 직접 연결되는 이유 제시

[JSON 스키마]
{{
  "watchlist": [
    {{
      "rank": 1,
      "name": "삼성전자",
      "market": "KOSPI",
      "sector": "반도체",
      "opinion": "관심",
      "reason": "string",
      "check_points": ["string", "string"],
      "related_news_titles": ["string"]
    }}
  ]
}}
""".strip()

    try:
        data = call_gemini_json(prompt)
        if isinstance(data, dict) and isinstance(data.get("watchlist"), list):
            result = []
            for idx, item in enumerate(data["watchlist"][:count], start=1):
                if not isinstance(item, dict):
                    continue
                check_points = item.get("check_points", [])
                if not isinstance(check_points, list):
                    check_points = []
                related_news_titles = item.get("related_news_titles", [])
                if not isinstance(related_news_titles, list):
                    related_news_titles = []

                opinion = clean_user_text(item.get("opinion", ""), "중립")
                if opinion not in ["관심", "중립", "주의"]:
                    opinion = "중립"

                result.append({
                    "rank": safe_int(item.get("rank", idx), default=idx, minimum=1),
                    "name": clean_user_text(item.get("name", ""), f"관심종목 {idx}"),
                    "market": clean_user_text(item.get("market", ""), "KOSPI/KOSDAQ"),
                    "sector": clean_user_text(item.get("sector", ""), "시장 전반"),
                    "opinion": opinion,
                    "reason": clean_user_text(item.get("reason", ""), "해당 뉴스 흐름과 연결되는 종목으로 참고할 수 있습니다."),
                    "check_points": [clean_user_text(v, "") for v in check_points if clean_user_text(v, "")][:3],
                    "related_news_titles": [clean_user_text(v, "") for v in related_news_titles if clean_user_text(v, "")][:3],
                })

            if result:
                return result
    except Exception:
        pass

    # fallback
    watchlist = []
    used = set()

    for _, row in df_top.iterrows():
        stocks = row["관련종목"] if isinstance(row["관련종목"], list) else []
        for stock in stocks:
            if stock in used:
                continue
            used.add(stock)
            watchlist.append({
                "rank": len(watchlist) + 1,
                "name": stock,
                "market": "KOSPI/KOSDAQ",
                "sector": row["관련업종"],
                "opinion": row["투자의견"],
                "reason": row["투자근거"],
                "check_points": [row["핵심시그널"]],
                "related_news_titles": [row["번역제목"]],
            })
            if len(watchlist) >= count:
                break
        if len(watchlist) >= count:
            break

    if not watchlist:
        watchlist = [
            {
                "rank": 1,
                "name": "삼성전자",
                "market": "KOSPI",
                "sector": "반도체",
                "opinion": "중립",
                "reason": "오늘 뉴스 흐름상 반도체와 기술 업종 전반을 점검할 필요가 있습니다.",
                "check_points": ["금리와 외국인 수급", "실적 기대치 변화"],
                "related_news_titles": [],
            }
        ]
    return watchlist

# ──────────────────────────────────────────
# 후보군 / 최종 선별
# ──────────────────────────────────────────
def select_candidate_pool(df_input, pool_size=10, min_domestic=3, min_global=3):
    domestic = df_input[df_input["국내외구분"] == "국내"].copy()
    global_df = df_input[df_input["국내외구분"] == "해외"].copy()

    selected = pd.concat([
        domestic.head(min_domestic),
        global_df.head(min_global)
    ]).drop_duplicates(subset=["row_id"])

    remain = df_input[~df_input["row_id"].isin(selected["row_id"])].copy()
    need = max(pool_size - len(selected), 0)

    if need > 0:
        selected = pd.concat([selected, remain.head(need)]).drop_duplicates(subset=["row_id"])

    return selected.sort_values(
        ["규칙점수", "발행일"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


def select_top_news(df_input, top_n=6, min_domestic=2, min_global=2):
    df_input = df_input.copy()
    df_input["이슈중복키"] = df_input["기사제목"].apply(make_issue_dedup_key)
    df_input = df_input.sort_values(
        ["중요도점수", "발행일"], ascending=[False, False], na_position="last"
    ).copy()

    df_unique = df_input.drop_duplicates(subset=["이슈중복키"], keep="first").copy()

    domestic = df_unique[df_unique["국내외구분"] == "국내"].head(min_domestic)
    global_df = df_unique[df_unique["국내외구분"] == "해외"].head(min_global)
    selected = pd.concat([domestic, global_df]).drop_duplicates(subset=["row_id"])

    remain = df_unique[~df_unique["row_id"].isin(selected["row_id"])].copy()
    need = max(top_n - len(selected), 0)

    if need > 0:
        selected = pd.concat([selected, remain.head(need)]).drop_duplicates(subset=["row_id"])

    selected = selected.sort_values(
        ["중요도점수", "발행일"], ascending=[False, False], na_position="last"
    ).head(top_n).reset_index(drop=True)

    selected["순위"] = selected.index + 1
    return selected

# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("경제 뉴스 브리프 시스템 시작")
    print(f"실행 시각: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')} KST")
    print(f"사용 모델: {GEMINI_MODEL}")
    print("=" * 60)

    # STEP 1. RSS 수집
    records = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])
            entries = parsed.entries[:MAX_ARTICLES_PER_FEED]

            for entry in entries:
                title = clean_text(entry.get("title", ""))
                link = entry.get("link", "")
                pub_raw = entry.get("published", "") or entry.get("pubDate", "") or entry.get("updated", "")
                summary = clean_text(entry.get("summary", "") or entry.get("description", ""))

                records.append({
                    "수집일시": datetime.now(KST).replace(tzinfo=None),
                    "출처": feed["source"],
                    "국내외구분": feed["region"],
                    "기본언어": feed["default_lang"],
                    "피드카테고리": feed["feed_hint"],
                    "기사제목": title,
                    "기사링크": link,
                    "기사설명": summary,
                    "발행일_원문": pub_raw,
                })

            print(f"  ✅ {feed['source']} | {feed['feed_hint']}: {len(entries)}건 수집")
        except Exception as e:
            print(f"  ⚠️ {feed['source']} | {feed['feed_hint']} 수집 실패: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("RSS 수집 결과가 비어 있습니다.")

    print(f"\n[STEP 1] RSS 수집 완료: {len(df)}건")

    # STEP 2. 정제
    df["기사제목"] = df["기사제목"].apply(clean_text)
    df["기사설명"] = df["기사설명"].apply(clean_text)
    df["제목정규키"] = df["기사제목"].apply(normalize_title_key)
    df["발행일"] = df["발행일_원문"].apply(parse_pubdate)
    df["발행일_표준"] = df["발행일"].apply(format_kst)

    df["기사설명"] = df.apply(
        lambda r: r["기사설명"] if len(r["기사설명"]) >= 20 else r["기사제목"],
        axis=1
    )

    df["언어"] = df.apply(
        lambda r: detect_language_safe(f"{r['기사제목']} {r['기사설명']}", default=r["기본언어"]),
        axis=1
    )

    df = df[
        (df["기사제목"].astype(str).str.len() > 0) &
        (df["기사링크"].astype(str).str.len() > 0)
    ].copy()

    before = len(df)

    df = df.sort_values("발행일", ascending=False, na_position="last").drop_duplicates(
        subset=["기사링크"], keep="first"
    ).copy()

    cluster_counts = df["제목정규키"].value_counts().to_dict()
    df["제목클러스터건수"] = df["제목정규키"].map(cluster_counts)

    df = df.drop_duplicates(subset=["제목정규키"], keep="first").copy()

    df["카테고리_규칙"] = df.apply(
        lambda r: guess_category(f"{r['피드카테고리']} {r['기사제목']} {r['기사설명']}"),
        axis=1
    )

    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1

    print(f"[STEP 2] 정제 완료: {before}건 → {len(df)}건")

    # STEP 3. 규칙 점수
    df["규칙점수"] = df.apply(score_news_row, axis=1)
    df = df.sort_values(["규칙점수", "발행일"], ascending=[False, False], na_position="last").copy()
    print("[STEP 3] 규칙 기반 점수 계산 완료")

    # STEP 4. 후보군
    df_candidates = select_candidate_pool(
        df,
        pool_size=CANDIDATE_POOL_SIZE,
        min_domestic=CANDIDATE_MIN_DOMESTIC,
        min_global=CANDIDATE_MIN_GLOBAL,
    )
    print(f"[STEP 4] 후보군 선별: {len(df_candidates)}건")

    # STEP 5. 기사 배치 분석
    print(f"[STEP 5] 배치 분석 시작: {len(df_candidates)}건 → Gemini 1회 호출")
    batch_result = analyze_articles_batch(df_candidates)

    result_map = {}
    if isinstance(batch_result, list):
        for pos, item in enumerate(batch_result, start=1):
            if not isinstance(item, dict):
                continue
            idx = safe_int(item.get("idx", pos), default=pos, minimum=1, maximum=len(df_candidates))
            if idx not in result_map:
                result_map[idx] = item

    analysis_results = []
    for i, (_, row) in enumerate(df_candidates.iterrows(), start=1):
        item = result_map.get(i)
        analysis_results.append(build_article_analysis(row.to_dict(), item=item))

    print("[STEP 5] Gemini 분석 완료")

    df_ai = pd.DataFrame(analysis_results)
    df_candidates = df_candidates.merge(df_ai, on="row_id", how="left")

    df_candidates["중요도점수"] = (
        df_candidates["규칙점수"].fillna(0) + df_candidates["AI중요도"].fillna(5) * 1.8
    ).round(2)

    # STEP 6. 최종 뉴스 선별
    df_top = select_top_news(
        df_candidates,
        top_n=TOP_NEWS_COUNT,
        min_domestic=FINAL_MIN_DOMESTIC,
        min_global=FINAL_MIN_GLOBAL,
    )
    print(f"[STEP 6] 최종 뉴스 선별: {len(df_top)}건")

    # STEP 7. 브리프 생성
    brief_data = generate_daily_brief(df_top)
    print("[STEP 7] 일일 브리프 생성 완료")

    # STEP 8. watchlist 생성
    watchlist = generate_watchlist(df_top, count=WATCHLIST_COUNT)
    print("[STEP 8] watchlist 생성 완료")

    # STEP 9. JSON 조립
    analysis_date = datetime.now(KST).strftime("%Y-%m-%d")
    generated_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

    top_news_list = []
    for _, row in df_top.iterrows():
        title = clean_user_text(row["번역제목"], row["기사제목"])
        description = clean_user_text(row["번역설명"], row["기사설명"])

        summary_lines = [
            clean_user_text(row["요약1"], ""),
            clean_user_text(row["요약2"], ""),
            clean_user_text(row["요약3"], ""),
        ]
        if len([s for s in summary_lines if s]) != 3:
            summary_lines = fallback_summary_lines(row["기사제목"], row["기사설명"], row["카테고리"])

        keywords = normalize_keywords(row["AI핵심키워드"], title=row["기사제목"], desc=row["기사설명"])
        related_stocks = row["관련종목"] if isinstance(row["관련종목"], list) else []

        top_news_list.append({
            "rank": int(row["순위"]),
            "region": row["국내외구분"],
            "classification": row["국내외구분"],
            "source": row["출처"],
            "title": title,
            "display_title": title,
            "original_title": row["기사제목"],
            "summary": summary_lines,
            "summary_text": " ".join(summary_lines),
            "description": description,
            "category": row["카테고리"],
            "importance": int(row["AI중요도"]),
            "market_impact": row["시장영향"],
            "related_sector": row["관련업종"],
            "related_stocks": related_stocks,
            "investment_view": row["투자의견"],
            "investment_reason": clean_user_text(row["투자근거"], ""),
            "key_signal": clean_user_text(row["핵심시그널"], ""),
            "keywords": keywords,
            "link": row["기사링크"],
            "published": row["발행일_표준"],
            "quality": {
                "title_status": row.get("제목상태", "unknown"),
                "summary_status": row.get("요약상태", "unknown"),
                "analysis_status": row.get("분석상태", "unknown"),
            }
        })

    output = {
        "generated_at": generated_at,
        "analysis_date": analysis_date,
        "project_overview": build_project_overview(),
        "brief": brief_data,
        "top_news": top_news_list,
        "watchlist": watchlist,
        "notice": "본 브리프는 정보 제공용이며 특정 종목의 매수·매도를 권유하지 않습니다."
    }

    # STEP 10. 저장
    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료! → {OUTPUT_PATH} 저장됨")
    print(f"   뉴스: {len(top_news_list)}건 / watchlist: {len(watchlist)}건")


if __name__ == "__main__":
    main()
