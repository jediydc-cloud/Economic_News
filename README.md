# 📈 Economic_News

매일 아침 자동으로 국내외 경제 뉴스를 수집·정리하고,  
투자 관점에서 핵심 이슈와 관심 종목을 도출해 `docs/data.json`으로 배포하는 프로젝트입니다.

이 저장소는 GitHub Actions + Gemini API를 사용하여  
**경제 뉴스 브리프**를 자동 생성합니다.

---

## 주요 기능

- 국내외 경제 뉴스 자동 수집
- 중복 제거 및 핵심 뉴스 선별
- 한국어 요약 및 투자 관점 해석
- 중요도 평가
- 관련 섹터/관련 종목 추출
- 관심 종목(Watchlist) 정리
- GitHub Pages / 웹사이트 위젯 연동

---

## 프로젝트 구조

```bash
Economic_News/
├─ .github/
│  └─ workflows/
│     └─ daily_report.yml
├─ docs/
│  ├─ data.json
│  └─ widget_embed.html
├─ main.py
├─ requirements.txt
└─ README.md
