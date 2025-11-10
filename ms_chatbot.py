# 💬 명신여자고등학교 Q&A 챗봇 (PDF 기반 RAG)
import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

# LangChain 및 Google GenAI 관련 모듈
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnablePassthrough

# --- 1. Gemini API 키 설정 ---
# NOTE: 실제 Streamlit 환경에서는 st.secrets["GOOGLE_API_KEY"]를 사용해야 합니다.
# 이 환경에서는 os.environ을 사용합니다.
try:
    if "GOOGLE_API_KEY" not in os.environ:
        # 이 부분은 실제 Streamlit 배포 시 st.secrets에서 가져오도록 수정 필요
        # st.error("⚠️ GOOGLE_API_KEY를 환경 변수 또는 Streamlit Secrets에 설정해주세요!")
        # st.stop()
        pass # 현재 실행 환경에서는 API Key가 이미 설정되어 있다고 가정합니다.
except Exception as e:
    # st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    # st.stop()
    pass

# --- 2. PDF 내용을 Document 객체로 변환 ---
# 첨부된 PDF의 내용을 텍스트로 추출하여 LangChain Document 객체 리스트로 만듭니다.
# 실제 RAG 구현에서는 Vector Store와 Embedding 모델이 필요하지만,
# 여기서는 제공된 텍스트가 매우 짧으므로 **Stuffing 방식**으로 단순화하여 구현합니다.
# (문서 전체를 프롬프트에 넣어 답변을 생성하는 방식)

# PDF의 모든 텍스트를 하나의 문자열로 결합 (소스에서 내용만 추출)
# 주의: 이 코드는 제공된 PDF 내용을 하드코딩한 것입니다. 실제 파일 처리 로직이 아닙니다.
pdf_content_text = """
명신여자고등학교의 위치는 대한민국 인천광약시 부평구 산곡동 부평구 원적로 260에 위치해 있다. 사립고등학교이다.
창립은 1970년에 했다. 문의 할 수 있는 전화번호는 032-502-3088.
교훈은 성실이다. 교목은 향나무. 교화는 장미이다. 학교 홈페이지는 https://msrose.icehs.kr/main.do이다.
교명(밝을 명(明), 새로울 신(新))은 밝고 새로우며, 광명하여 날로 새로워지는 진보 발전하는 학교를 의미한다.
교표는 둥근 원(우주, 성실, 신의, 봉사의 이념, 진리 불변과 순환, 완전한 하나 상징)과 정사각형(안정, 견고함 상징)으로 구성되어 있고, 붉은 색은 생명과 사랑을 의미하며, 1971 숫자는 설립년도를 의미한다.
교기는 발전하는 역사와 미래를 정점으로 상징하며, 초록색 바탕은 높은 기상과 부흥의 의지, 무한한 가능성을 의미한다. 하단에는 학교명이 금색으로 자수되어 있다.
역대 이사장은 이정월, 강종락, 강지원이다.
역대 교장은 이주환, 최원택, 이창봉, 김용오, 리범직, 권유상, 조규배, 한병옥, 천민수, 윤동춘, 이남정, 강인수, 이영자, 이종혁, 권용석, 윤인리 순이다.
2025학년도 입학생 3개년간 교육과정 편성표에 포함된 과목 정보:
- **국어:** 공통국어1, 공통국어2, 독서와 작문, 문학 (총이수 14단위, 필수 10단위)
- **수학:** 공통수학1, 공통수학2, 대수, 미적분 1 (총이수 13단위, 필수 10단위)
- **영어:** 공통영어1, 공통영어2, 영어 1, 영어 II (총이수 16단위, 필수 10단위)
- **체육:** 체육 1, 체육 IⅡ, 스포츠 문화, 스포츠 과학, 스포츠 생활 1, 스포츠 생활 Ⅱ (총이수 10단위, 필수 10단위)
- **예술:** 음악, 미술, 음악 감상과 비평, 미술 감상과 비평 (총이수 10단위, 필수 10단위)
- **사회(역사/도덕 포함):** 한국사 1, 한국사 Ⅱ, 통합사회 1, 통합사회 II (총이수 12단위, 필수 10단위)
- **과학:** 통합과학 1, 통합과학 II, 과학탐구실험 1, 과학탐구실험 II (총이수 8단위, 필수 12단위)
- **기술·가정/정보:** 정보, 지식 재산 일반 (총이수 8단위, 필수 0단위)
- **교양:** 진로와 직업, 생태와 환경, 인간과 철학, 인간과 심리, 교육의 이해, 보건 (선택)
- **제2외국어/한문:** 중국 문화, 일본 문화, 언어생활과 한자, 중국어 1, 일본어 1, 한문 (융합/일반 선택)
- **선택 과목 (일부):** 기하, 세계 문화와 영어, 기초 체육 전공 실기, 미술과 매체, 음악과 문화, 정보과학, 정치, 윤리와 사상, 역학과 에너지, 물질과 에너지, 세포와 물질대사, 지구시스템과학, 언어생활 탐구 등 다수 과목.
총 이수 단위는 학기별 32단위 (6학기), 총 **192단위**이다. (창의적 체험활동 18단위 포함).
"""

# 하나의 큰 Document로 만듦 (Stuffing 방식)
retrieved_docs = [Document(page_content=pdf_content_text, metadata={"source": "명신여고 소개 PDF"})]

# 간단한 인메모리 검색기 (항상 모든 문서를 반환)
# 실제 RAG에서는 Vector Store 기반의 Retriever를 사용합니다.
class SimpleInMemoryRetriever:
    def __init__(self, documents):
        self.documents = documents

    def get_relevant_documents(self, query):
        # 쿼리에 관계없이 모든 문서를 반환 (Stuffing 방식의 단순화)
        return self.documents

retriever = SimpleInMemoryRetriever(retrieved_docs)


# --- 3. LLM 및 프롬프트 설정 (캐시) ---
@st.cache_resource(show_spinner="🤖 Q&A 챗봇 모델 및 지식 기반 로딩 중...")
def get_qa_chain(selected_model):
    """
    RAG 기반 Q&A 체인을 생성합니다.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.0, # Q&A는 창의성보다 정확성이 중요하므로 0.0 설정
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 API 키가 유효한지, 모델 이름이 올바른지 확인해보세요.")
        st.stop()

    # 1. 문서 결합 체인 (Document Combination Chain)
    # 검색된 문서를 바탕으로 답변을 생성하는 프롬프트
    qa_system_prompt = (
        "당신은 명신여자고등학교의 정보를 제공하는 친절하고 정확한 AI 어시스턴트 '제미나이'입니다. "
        "항상 한국어와 존댓말을 사용하며, 제공된 **다음 정보(context)**만을 사용하여 사용자의 질문에 답변하세요. "
        "만약 정보에 없는 내용이라면 '죄송하지만 제공된 자료에는 해당 내용이 없습니다.'라고 답하세요. "
        "답변 시 이모지를 적절히 사용해 주세요. 🤖\n\n"
        "**Context:**\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 2. 검색 체인 (Retrieval Chain)
    # 검색된 문서와 사용자 입력을 결합 체인에 전달하는 전체 체인
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- 4. Streamlit UI 설정 ---

st.header("명신여자고등학교 Q&A 챗봇 🏫")
st.info("첨부된 PDF 파일을 기반으로 명신여고에 대해 질문하세요.")

# 채팅 기록을 Streamlit의 세션 상태(session_state)에 저장
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# 모델 선택
option = st.selectbox("Select Gemini Model",
    ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"),
    index=0,
    help="가장 빠르고 효율적인 2.5 Flash 모델을 추천합니다."
)

# 선택된 모델로 LLM 체인 가져오기
retrieval_chain = get_qa_chain(option)

# 대화 기록을 관리하는 Runnable 생성
# LangChain에서는 `RunnableWithMessageHistory`를 사용하여 대화 기록을 관리합니다.
conversational_retrieval_chain = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history", # history 대신 qa_prompt에 설정한 chat_history 키 사용
)

# --- 5. 채팅 UI 로직 ---

# 첫 방문 시 환영 메시지 추가
if not chat_history.messages:
    chat_history.add_ai_message("안녕하세요! 저는 명신여고 PDF 기반 Q&A 챗봇 '제미나이'입니다. 😊 학교 위치, 교훈, 교육과정 등에 대해 물어보세요!")

# 이전 대화 기록 모두 출력
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 사용자 입력 받기
if prompt_message := st.chat_input("명신여고에 대해 질문하세요..."):
    # 사용자가 입력한 메시지 출력
    st.chat_message("human").write(prompt_message)
    
    # AI 응답 생성 및 출력
    with st.chat_message("ai"):
        with st.spinner("정보를 검색하고 답변을 생성 중..."):
            # config: session_id는 아무 값이나 넣어도 chat_history를 사용하도록 설정됨
            config = {"configurable": {"session_id": "any_id"}}
            
            # RAG 체인 실행
            # 반환되는 결과는 {'answer': '...', 'context': [...]} 형태입니다.
            response_data = conversational_retrieval_chain.invoke(
                {"input": prompt_message},
                config
            )
            
            # 답변 출력
            st.write(response_data["answer"])
            
            # (선택 사항) 검색된 문서(context)를 보여줌
            # with st.expander("🔍 검색된 정보 (Context)"):
            #     st.json(response_data["context"])
