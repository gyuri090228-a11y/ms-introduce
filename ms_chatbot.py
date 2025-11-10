import json

# 1. 명신여자고등학교 소개 정보를 딕셔너리(JSON 유사 구조)로 구조화
#    Key-Value 쌍으로 정보를 저장하여 특정 항목 접근을 용이하게 합니다.
myungshin_info = {
    "기본_정보": {
        "위치": "대한민국 인천광역시 부평구 산곡동 부평구 원적로 260",
        "구분": "사립고등학교",
        "창립_연도": "1970년",
        "전화번호": "032-502-3088",
        "홈페이지": "https://msrose.icehs.kr/main.do",
        "교훈": "성실",
        "교목": "향나무",
        "교화": "장미"
    },
    "교명_의미": {
        "밝을_명(明)": "긍정적 심성, 마음과 뜻도 통하며, 최고, 깨끗하며, 광명의 심성을 내포",
        "새로울_신(新)": "정체, 침체, 정지 아니하고, 진보, 발전, 향상을 의미",
        "전체_의미": "밝고 새로우며, 광명하여 날로 새로워지는 진보 발전하는 학교"
    },
    "상징": {
        "교표": {
            "둥근_원": "우주를 성실, 신의, 봉사의 이념을 구심점으로 진리의 불변과 순환의 항상성을 지닌 완전한 하나로 되어 가는 것을 상징",
            "정사각형": "안정을 표상하고 견고함을 상징",
            "붉은_색": "생명과 사랑을 의미",
            "1971_숫자": "학교의 설립년도를 의미"
        },
        "교기": {
            "상징": "발전하는 보라의 역사와 미래를 정점으로 상징",
            "초록색_바탕": "높은 기상과 부흥의 의지, 그리고 무한한 가능성을 의미"
        }
    },
    "교육과정_요약": {
        "국어": {"총이수_단위": 14, "필수이수_단위": 10},
        "수학": {"총이수_단위": 13, "필수이수_단위": 10},
        "영어": {"총이수_단위": 16, "필수이수_단위": 10},
        "체육": {"총이수_단위": 10, "필수이수_단위": 10},
        "예술": {"총이수_단위": 10, "필수이수_단위": 10},
        "사회(역사/도덕_포함)": {"총이수_단위": 12, "필수이수_단위": 10},
        "과학": {"총이수_단위": 8, "필수이수_단위": 12},
        "기술·가정/정보": {"총이수_단위": 8, "필수이수_단위": 0}
    },
    "총_이수_단위": {
        "학기별_총_단위": "32단위 x 6학기 = 192단위 (정상)",
        "학년별_총_단위": "64단위 x 3학년 = 192단위 (정상)"
    }
}

# 2. 사용자 요청에 따라 정보를 '생성'하여 출력하는 함수
def generate_info(query):
    """
    사용자의 요청(query)에 따라 myungshin_info 딕셔너리에서 적절한 정보를 추출 및 조합하여 반환합니다.
    (간단한 키워드 매칭을 통해 생성형 AI의 역할을 대신합니다.)
    """
    if "위치" in query or "주소" in query or "기본" in query:
        # 기본 정보 섹션에서 정보 조합
        location = myungshin_info['기본_정보']['위치']
        establishment = myungshin_info['기본_정보']['창립_연도']
        return f"🏫 명신여자고등학교의 기본 정보입니다:\n위치: {location}\n구분: {myungshin_info['기본_정보']['구분']}\n창립 연도: {establishment}"

    elif "교명" in query or "이름" in query or "뜻" in query:
        # 교명 의미 섹션에서 정보 조합
        bright = myungshin_info['교명_의미']['밝을_명(明)']
        new = myungshin_info['교명_의미']['새로울_신(新)']
        full_meaning = myungshin_info['교명_의미']['전체_의미']
        return f"✨ 명신(明新) 교명의 의미:\n- 밝을 명(明): {bright}\n- 새로울 신(新): {new}\n- 전체 의미: {full_meaning}"

    elif "교훈" in query or "상징" in query or "교목" in query or "교화" in query:
        # 기본 정보 및 상징 섹션에서 정보 조합
        symbol_info = (
            f"🌲 교목: {myungshin_info['기본_정보']['교목']}\n"
            f"🌹 교화: {myungshin_info['기본_정보']['교화']}\n"
            f"🙏 교훈: {myungshin_info['기본_정보']['교훈']}\n"
            f"⚫ 교표의 둥근 원은 {myungshin_info['상징']['교표']['둥근_원']}\n"
            f"🟩 교기의 초록색 바탕은 {myungshin_info['상징']['교기']['초록색_바탕']}"
        )
        return f"💡 학교 상징 정보입니다:\n{symbol_info}"

    elif "교육과정" in query or "이수 단위" in query or "학점" in query:
        # 교육과정 요약 및 총 이수 단위 정보 조합
        summary = "📜 2025학년도 입학생 주요 교과 영역별 이수 단위 (총이수/필수이수):\n"
        for subject, units in myungshin_info['교육과정_요약'].items():
            summary += f"- **{subject}**: {units['총이수_단위']} 단위 / {units['필수이수_단위']} 단위\n"
        
        summary += (
            f"\n🎓 **총 이수 단위**: \n"
            f"학기별 총 이수 단위: {myungshin_info['총_이수_단위']['학기별_총_단위']}\n"
            f"학년별 총 이수 단위: {myungshin_info['총_이수_단위']['학년별_총_단위']}"
        )
        return summary
        
    elif "모든" in query or "전체" in query:
        # 모든 정보 JSON 형태로 출력
        return "📄 **명신여자고등학교의 모든 정보 (JSON 형식)**:\n" + json.dumps(myungshin_info, indent=4, ensure_ascii=False)

    else:
        # 요청에 맞는 정보가 없을 경우 안내
        return "🤷‍♀️ 죄송해요. '위치', '교명', '교훈', '교육과정' 등의 키워드를 포함하여 다시 질문해 주시면 해당 정보를 생성해 드릴게요."

# 3. 코드 실행 예시: 사용자가 궁금한 내용을 입력
print("--- [코드 실행 시작] ---\n")

# 예시 1: 기본 정보 요청
user_query_1 = "학교의 위치와 창립 연도를 알려줘."
print(f"**[사용자 요청]**: {user_query_1}")
response_1 = generate_info(user_query_1)
print(f"**[AI 응답]**:\n{response_1}\n")
print("-" * 30)

# 예시 2: 교명의 의미 요청
user_query_2 = "명신이라는 이름의 뜻이 뭐야?"
print(f"**[사용자 요청]**: {user_query_2}")
response_2 = generate_info(user_query_2)
print(f"**[AI 응답]**:\n{response_2}\n")
print("-" * 30)

# 예시 3: 교육과정 요약 요청
user_query_3 = "주요 교과 영역별 이수 단위를 요약해 줘."
print(f"**[사용자 요청]**: {user_query_3}")
response_3 = generate_info(user_query_3)
print(f"**[AI 응답]**:\n{response_3}\n")

print("\n--- [코드 실행 종료] ---")
