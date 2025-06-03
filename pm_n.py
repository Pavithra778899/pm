import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px
import time
import uuid
import retrying

# --- Snowflake/Cortex Configuration ---
HOST = "QNWFESR-LKB66742.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000
CORTEX_SEARCH_SERVICES = None  # Set dynamically
SEMANTIC_MODEL = '@"AI"."DWH_MART"."PROPERTY_MANAGEMENT"/property_management (1).yaml'

# --- Model Options ---
MODELS = [
    "mistral-large",
    "snowflake-arctic",
    "llama3-70b",
    "llama3-8b",
]

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Cortex AI-Property Management Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Session State Initialization ---
for key, value in {
    "authenticated": False,
    "username": "",
    "password": "",
    "CONN": None,
    "snowpark_session": None,
    "chat_history": [],
    "messages": [],
    "debug_mode": False,
    "last_suggestions": [],
    "chart_x_axis": None,
    "chart_y_axis": None,
    "chart_type": "Bar Chart",
    "current_query": None,
    "current_results": None,
    "current_sql": None,
    "current_summary": None,
    "service_metadata": [{"name": "", "search_column": ""}],
    "selected_cortex_search_service": "",
    "model_name": "mistral-large",
    "num_retrieved_chunks": 100,
    "num_chat_messages": 10,
    "use_chat_history": True,
    "clear_conversation": False,
    "show_selector": False,
    "show_greeting": True,
    "show_about": False,
    "show_help": False,
    "show_history": False,
    "query": None,
    "previous_query": None,
    "previous_sql": None,
    "previous_results": None,
    "show_sample_questions": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- CSS Styling ---
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow: hidden !important;
}
[data-testid="stChatMessageContent"] {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow: auto !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.copy-button, [data-testid="copy-button"], [title="Copy to clipboard"], [data-testid="stTextArea"] {
    display: none !important;
}
.dilytics-logo {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    width: 150px;
    height: auto;
}
.fixed-header {
    position: fixed;
    top: 0;
    left: 20px;
    right: 0;
    z-index: 999;
    background-color: #ffffff;
    padding: 10px;
    text-align: center;
    pointer-events: none;
}
.fixed-header a {
    pointer-events: none !important;
    text-decoration: none !important;
    color: inherit !important;
    cursor: default !important;
}
.stApp {
    padding-top: 100px;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background-color: #29B5E8 !important;
    color: white !important;
    font-weight: bold !important;
    width: 100% !important;
    border-radius: 0px !important;
    margin: 0 !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
}
[data-testid="stSidebar"] [data-testid="stButton"][data-testid="Clear conversation"] > button,
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="About"] > button,
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="Help & Documentation"] > button,
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="History"] > button,
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="Sample Questions"] > button {
    background-color: #28A745 !important;
    color: white !important;
    font-weight: normal !important;
    border: 1px solid #28A745 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Add Logo ---
if st.session_state.authenticated:
    st.markdown(
        '<img src="https://raw.githubusercontent.com/nkumbala129/30-05-2025/main/Dilytics_logo.png" class="dilytics-logo">',
        unsafe_allow_html=True
    )

# --- Utility Functions ---
def stream_text(text: str, chunk_size: int = 10, delay: float = 0.02):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(delay)

def submit_maintenance_request(property_id: str, tenant_name: str, issue_description: str):
    try:
        request_id = str(uuid.uuid4())
        insert_query = """
        INSERT INTO MAINTENANCE_REQUESTS 
        (REQUEST_ID, PROPERTY_ID, TENANT_NAME, ISSUE_DESCRIPTION, SUBMITTED_AT, STATUS)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP(), 'PENDING')
        """
        session.sql(insert_query).execute([request_id, property_id, tenant_name, issue_description])
        return True, f"📝 Maintenance request submitted successfully! Request ID: {request_id}"
    except Exception as e:
        return False, f"Failed to submit maintenance request: {str(e)}"

def start_new_conversation():
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.current_query = None
    st.session_state.current_results = None
    st.session_state.current_sql = None
    st.session_state.current_summary = None
    st.session_state.chart_x_axis = None
    st.session_state.chart_y_axis = None
    st.session_state.chart_type = "Bar Chart"
    st.session_state.last_suggestions = []
    st.session_state.clear_conversation = False
    st.session_state.show_greeting = True
    st.session_state.query = None
    st.session_state.show_history = False
    st.session_state.previous_query = None
    st.session_state.previous_sql = None
    st.session_state.previous_results = None
    st.rerun()

def init_service_metadata():
    global CORTEX_SEARCH_SERVICES
    try:
        services = session.sql('SHOW CORTEX SEARCH SERVICES IN SCHEMA "AI"."DWH_MART";').collect()
        if not services:
            st.error("No Cortex Search Services found in AI.DWH_MART. Please create a service.")
            CORTEX_SEARCH_SERVICES = None
            st.session_state.service_metadata = [{"name": "", "search_column": ""}]
            st.session_state.selected_cortex_search_service = ""
            return
        service_name = None
        for svc in services:
            svc_name = svc["name"]
            full_name = f'"AI"."DWH_MART"."{svc_name}"'
            desc_result = session.sql(f'DESC CORTEX SEARCH SERVICE {full_name}').collect()
            if desc_result:
                service_name = full_name
                break
        if not service_name:
            st.error("No valid Cortex Search Service found in AI.DWH_MART.")
            CORTEX_SEARCH_SERVICES = None
            st.session_state.service_metadata = [{"name": "", "search_column": ""}]
            st.session_state.selected_cortex_search_service = ""
            return
        CORTEX_SEARCH_SERVICES = service_name
        desc_result = session.sql(f'DESC CORTEX SEARCH SERVICE {service_name}').collect()
        svc_search_col = desc_result[0]["search_column"] if desc_result else ""
        st.session_state.service_metadata = [{"name": service_name, "search_column": svc_search_col}]
        st.session_state.selected_cortex_search_service = service_name
    except Exception as e:
        st.error(f"Failed to load Cortex Search Service: {str(e)}")
        CORTEX_SEARCH_SERVICES = None
        st.session_state.service_metadata = [{"name": "", "search_column": ""}]
        st.session_state.selected_cortex_search_service = ""

def query_cortex_search_service(query: str) -> str:
    try:
        if not st.session_state.selected_cortex_search_service:
            st.error("No Cortex Search Service selected. Please review your service configuration.")
            return ""
        db, schema = session.get_current_database(), session.get_current_schema()
        root = Root(session)
        service_name = st.session_state.selected_cortex_search_service.split('.')[-1].strip('"')
        cortex_search_service = (
            root.databases[db]
            .schemas[schema]
            .cortex_search_services[service_name]
        )
        desc_result = session.sql(f'DESC CORTEX SEARCH SERVICE {st.session_state.selected_cortex_search_service}').collect()
        if not desc_result:
            st.error(f"Cortex Search Service {st.session_state.selected_cortex_search_service} does not exist.")
            return ""
        columns = [row["search_column"] for row in desc_result]
        context_documents = cortex_search_service.search(
            query, columns=columns or [], limit=st.session_state.num_retrieved_chunks
        )
        results = context_documents.results
        search_col = st.session_state.service_metadata[0]["search_column"] or columns[0] if columns else ""
        if not search_col:
            st.warning("No search column defined for Cortex Search Service.")
            return ""
        context_str = ""
        for i, r in enumerate(results):
            context_str += f"Context document {i+1}: {r.get(search_col, '')}\n\n"
        return context_str
    except Exception as e:
        st.error(f"Error querying Cortex Search Service: {str(e)}")
        return ""

def get_chat_history():
    start_index = max(0, len(st.session_state.chat_history) - st.session_state.num_chat_messages)
    return st.session_state.chat_history[start_index:len(st.session_state.chat_history) - 1]

def make_chat_history_summary(chat_history: List[Dict], question: str) -> str:
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    prompt = f"""
[INST]
You are an AI assistant specializing in property management queries. Rewrite the latest user question into a clear, standalone query by incorporating relevant context from the chat history. Ensure the rewritten query is precise, complete, and reflects the user's intent, avoiding ambiguity. Focus on property-related terms (e.g., leases, tenants, rent, occupancy).

Examples:
- Chat history: 
  user: What is the total number of leases?
  assistant: There are 150 leases.
  user: by state
  Rewritten query: What is the total number of leases by state?
- Chat history:
  user: List properties with high occupancy.
  assistant: Properties with occupancy above 90% are listed.
  user: for last year
  Rewritten query: List properties with high occupancy for last year.

Input format:
<chat_history>
{chat_history_str}
</chat_history>
<question>
{question}
</question>

Output only the rewritten standalone query.
[/INST]
"""
    summary = complete(st.session_state.model_name, prompt.strip())
    if st.session_state.debug_mode:
        st.sidebar.text_area("Resolved Question", summary.replace("$", "\\$"), height=150)
    return summary.strip()

def create_prompt(user_question: str) -> str:
    chat_history_str = ""
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context = query_cortex_search_service(question_summary)
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        else:
            prompt_context = query_cortex_search_service(user_question)
    else:
        prompt_context = query_cortex_search_service(user_question)
        chat_history = []
    
    if not prompt_context.strip():
        return complete(st.session_state.model_name, user_question)
    
    prompt = f"""
[INST]
You are a property management AI assistant. Provide a concise, accurate answer to the user's question using the provided context and chat history. Focus on delivering relevant insights about properties, leases, tenants, or occupancy.

<chat_history>
{chat_history_str}
</chat_history>
<context>
{prompt_context}
</context>
<question>
{user_question}
</question>
[/INST]
Answer:
"""
    return complete(st.session_state.model_name, prompt)

def get_user_questions(limit: int = 10) -> List[str]:
    user_questions = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    return user_questions[-limit:][::-1]

# --- Main Application Logic ---
if not st.session_state.authenticated:
    st.title("Welcome to Snowflake Cortex AI")
    st.markdown("Please login to interact with your data")
    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user="CORTEX",
                password="Dilytics@12345",
                account="QNWFESR-LKB66742",
                host=HOST,
                port=443,
                warehouse="COMPUTE_WH",
                role="ACCOUNTADMIN",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn
            snowpark_session = Session.builder.configs({"connection": conn}).create()
            st.session_state.snowpark_session = snowpark_session
            with conn.cursor() as cur:
                cur.execute(f"USE DATABASE {DATABASE}")
                cur.execute(f"USE SCHEMA {SCHEMA}")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")
            st.session_state.authenticated = True
            st.success("Authentication successful! Redirecting...")
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
else:
    session = st.session_state.snowpark_session
    root = Root(session)
    init_service_metadata()

    def run_snowflake_query(query: str) -> Optional[pd.DataFrame]:
        try:
            if not query:
                return None
            df = session.sql(query)
            data = df.collect()
            if not data:
                return None
            columns = df.schema.names
            result_df = pd.DataFrame(data, columns=columns)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            return None

    def is_structured_query(query: str) -> bool:
        structured_patterns = [
            r'\b(count|number|total|sum|avg|max|min|how many|which|show|list|group by|order by)\b',
            r'\b(property|properties|tenant|tenants|lease|leases|rent|occupancy|maintenance|billing|payment)\b',
            r'\b(by state|by city|by property|by year|by month)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

    def is_complete_query(query: str) -> bool:
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str) -> bool:
        summarize_patterns = [r'\b(summarize|summary|condense)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str) -> bool:
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|queries)\b.*\b(ask|can i ask)\b',
            r'\b(give me|show me|list)\b.*\b(questions|examples)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def is_greeting_query(query: str) -> bool:
        greeting_patterns = [
            r'^\s*(hi|hello|hey|greetings)\s*$',
            r'\bhow are you\b',
            r'\bhow’s it going\b',
            r'\bwhat’s up\b',
            r'\bgood (morning|afternoon|evening)\b',
            r'\bthank(s| you)\b',
            r'\bwho are you\b',
            r'\bwhat can you do\b',
            r'\bwhat do you offer\b',
            r'\bhow can you help\b',
            r'\bwhat can i get from you\b',
            r'\bwhat can i display\b',
            r'\bhelp\b',
            r'\bstart over\b',
            r'\bwhat is this (app|assistant|tool)\b',
            r'\btell me about yourself\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in greeting_patterns)

    def complete(model: str, prompt: str) -> Optional[str]:
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            st.error(f"❌ COMPLETE Function Error: {str(e)}")
            return None

    def summarize(text: str) -> Optional[str]:
        try:
            text = text.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query).collect()
            return result[0]["SUMMARY"]
        except Exception as e:
            st.error(f"❌ SUMMARIZE Function Error: {str(e)}")
            return None

    def parse_sse_response(response_text: str) -> List[Dict]:
        events = []
        lines = response_text.strip().split("\n")
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if data_str != "[DONE]":
                    try:
                        data_json = json.loads(data_str)
                        current_event["data"] = data_json
                        events.append(current_event)
                        current_event = {}
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse SSE data: {str(e)}")
        return events

    def process_sse_response(response: List[Dict], is_structured: bool) -> Tuple[str, List[str]]:
        sql = ""
        search_results = []
        if not response:
            return sql, search_results
        for event in response:
            if event.get("event") == "message.delta" and "data" in event:
                delta = event["data"].get("delta", {})
                content = delta.get("content", [])
                for item in content:
                    if item.get("type") == "tool_results":
                        tool_results = item.get("tool_results", {})
                        if "content" in tool_results:
                            for result in tool_results["content"]:
                                if result.get("type") == "json":
                                    result_data = result.get("json", {})
                                    if is_structured and "sql" in result_data:
                                        sql = result_data.get("sql", "")
                                    elif not is_structured and "searchResults" in result_data:
                                        search_results = [sr["text"] for sr in result_data["searchResults"]]
        return sql.strip(), search_results

    @retrying.retry(
        stop_max_attempt_number=3,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000
    )
    def snowflake_api_call(query: str, is_structured: bool = False) -> List[Dict]:
        payload = {
            "model": st.session_state.model_name,
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if is_structured:
            payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
            payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
        else:
            if not st.session_state.selected_cortex_search_service:
                raise Exception("No Cortex Search Service configured.")
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": st.session_state.selected_cortex_search_service, "max_results": st.session_state.num_retrieved_chunks}}
        try:
            resp = requests.post(
                url=f"https://{HOST}{API_ENDPOINT}",
                json=payload,
                headers={
                    "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
                    "Content-Type": "application/json",
                },
                timeout=API_TIMEOUT // 1000
            )
            if resp.status_code < 400:
                if not resp.text.strip():
                    raise Exception("API returned an empty response.")
                return parse_sse_response(resp.text)
            else:
                raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"❌ API Request Error: {str(e)}")
            raise

    def summarize_unstructured_answer(answer: str) -> str:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|")\s', answer)
        return "\n".join(f"• {sent.strip()}" for sent in sentences[:6] if sent.strip())

    def suggest_sample_questions(query: str) -> List[str]:
        try:
            prompt = (
                f"The user asked: '{query}'. Generate 3-5 clear, concise sample questions related to properties, leases, tenants, rent, or occupancy metrics. "
                f"Format as a numbered list."
            )
            response = complete(st.session_state.model_name, prompt)
            if response:
                questions = []
                for line in response.split("\n"):
                    line = line.strip()
                    if re.match(r'^\d+\.\s*.+', line):
                        question = re.sub(r'^\d+\.\s*', '', line)
                        questions.append(question)
                return questions[:5]
            else:
                return [
                    "What are the total number of active leases?",
                    "What is the average rent collected per tenant?",
                    "Total number of properties?",
                    "What’s the total rental income by property?",
                    "Which tenants have pending rent payments?"
                ]
        except Exception as e:
            return [
                "What are the total number of active leases?",
                "What is the average rent collected per tenant?",
                "Total number of properties?",
                "What’s the total rental income by property?",
                "Which tenants have pending rent payments?"
            ]

    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart", query: str = ""):
        try:
            if df is None or df.empty or len(df.columns) < 2:
                st.warning("No valid data available for visualization.")
                if st.session_state.debug_mode:
                    st.sidebar.warning(f"Chart Data Issue: df={df}, columns={df.columns if df is not None else 'None'}")
                return
            query_lower = query.lower()
            if re.search(r'\b(county|jurisdiction)\b', query_lower):
                default_data = "Pie Chart"
            elif re.search(r'\b(month|year|date)\b', query_lower):
                default_data = "Line Chart"
            else:
                default_data = "Bar Chart"
            all_cols = list(df.columns)
            col1, col2, col3 = st.columns(3)
            x_col = col1.selectbox("X axis", all_cols, index=0, key=f"{prefix}_x")
            remaining_cols = [c for c in all_cols if c != x_col]
            y_col = col2.selectbox("Y axis", remaining_cols, index=0, key=f"{prefix}_y")
            chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
            chart_type = col3.selectbox("Chart Type", chart_options, index=chart_options.index(default_data), key=f"{prefix}_type")
            if st.session_state.debug_mode:
                st.sidebar.text_area("Chart Config", f"X: {x_col}, Y: {y_col}, Type: {chart_type}", height=100)
            if df[x_col].nunique() < 1 or df[y_col].empty:
                st.warning("Insufficient or invalid data for selected axes.")
                return
            if chart_type != "Histogram Chart" and not pd.api.types.is_numeric_dtype(df[y_col]):
                st.warning("Y-axis must be numeric for this chart type.")
                return
            st.markdown(f"### 📊 {chart_type}")
            if chart_type == "Line Chart":
                fig = px.line(df, x=x_col, y=y_col, title=chart_type)
                st.plotly_chart(fig, key=f"{prefix}_line")
            elif chart_type == "Bar Chart":
                fig = px.bar(df, x=x_col, y=y_col, title=chart_type)
                st.plotly_chart(fig, key=f"{prefix}_bar")
            elif chart_type == "Pie Chart":
                fig = px.pie(df, names=x_col, values=y_col, title=chart_type)
                st.plotly_chart(fig, key=f"{prefix}_pie")
            elif chart_type == "Scatter Chart":
                fig = px.scatter(df, x=x_col, y=y_col, title=chart_type)
                st.plotly_chart(fig, key=f"{prefix}_scatter")
            elif chart_type == "Histogram Chart":
                fig = px.histogram(df, x=x_col, title=chart_type)
                st.plotly_chart(fig, key=f"{prefix}_hist")
        except Exception as e:
            st.error(f"❌ Error generating chart: {str(e)}")
            if st.session_state.debug_mode:
                st.sidebar.error(f"Chart Error Details: {str(e)}")

    def toggle_about():
        st.session_state.show_about = not st.session_state.show_about
        st.session_state.show_help = False
        st.session_state.show_history = False

    def toggle_help():
        st.session_state.show_help = not st.session_state.show_help
        st.session_state.show_about = False
        st.session_state.show_history = False

    def toggle_history():
        st.session_state.show_history = not st.session_state.show_history
        st.session_state.show_about = False
        st.session_state.show_help = False

    # --- Sidebar ---
    with st.sidebar:
        logo_url = "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg"
        st.image(logo_url, width=250)
        if st.button("Clear conversation", key="clear_conversation"):
            start_new_conversation()
        if CORTEX_SEARCH_SERVICES:
            st.selectbox(
                "Select Cortex Search Service:",
                [CORTEX_SEARCH_SERVICES],
                index=0,
                key="selected_cortex_search_service"
            )
        else:
            st.warning("⚠️ No Cortex Search Service available.")
        st.toggle("Debug", key="debug_mode")
        with st.expander("Advanced options"):
            st.selectbox("Select model:", MODELS, key="model_name")
            st.number_input(
                "Select number of context chunks",
                value=100,
                key="num_retrieved_chunks",
                min_value=1,
                max_value=400
            )
            st.number_input(
                "Select number of messages to use in chat history",
                value=10,
                key="num_chat_messages",
                min_value=1,
                max_value=100
            )
        if st.button("Sample Questions", key="sample_questions"):
            st.session_state.show_sample_questions = not st.session_state.get("show_sample_questions", False)
        if st.session_state.get("show_sample_questions", False):
            st.markdown("### Sample Questions")
            sample_questions = [
                "What is Property Management?",
                "Total number of properties currently occupied?",
                "What is the number of properties by occupancy status?",
                "What is the number of properties currently leased?",
                "What is the average rent collected per tenant?",
                "Total number of properties?",
                "What’s the total rental income by property?",
                "Which tenants have pending rent payments?"
            ]
            for sample in sample_questions:
                if st.button(sample, key=f"sidebar_{sample}"):
                    st.session_state.query = sample
                    st.session_state.show_greeting = False
        with st.expander("Submit Maintenance Request"):
            st.markdown("### Submit a Maintenance Request")
            property_id = st.text_input("Property ID", key="maint_property_id")
            tenant_name = st.text_input("Tenant Name", key="maint_tenant_name")
            issue_description = st.text_area("Issue Description", key="maint_issue_description")
            if st.button("Submit Request", key="submit_maintenance_request"):
                if not property_id or not tenant_name or not issue_description:
                    st.error("❌ Please fill in all fields to submit a maintenance request.")
                else:
                    success, message = submit_maintenance_request(property_id, tenant_name, issue_description)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        st.markdown("---")
        if st.button("History", key="history"):
            toggle_history()
        if st.session_state.show_history:
            st.markdown("### Recent Questions")
            user_questions = get_user_questions(limit=10)
            if not user_questions:
                st.write("No questions in history yet.")
            else:
                for idx, question in enumerate(user_questions):
                    if st.button(question, key=f"history_{idx}"):
                        st.session_state.query = question
                        st.session_state.show_greeting = False
        if st.button("About", key="about"):
            toggle_about()
        if st.session_state.show_about:
            st.markdown("### About")
            st.write(
                "This application uses **Snowflake Cortex Analyst** to interpret "
                "your natural language questions and generate data insights for property management."
            )
        if st.button("Help & Documentation", key="help"):
            toggle_help()
        if st.session_state.show_help:
            st.markdown("### Help & Documentation")
            st.write(
                "- [User Guide](https://docs.snowflake.com/en/guides-overview-ai-features)\n"
                "- [Snowflake Cortex Analyst Docs](https://docs.snowflake.com/)\n"
                "- [Contact Support](https://www.snowflake.com/en/support/)"
            )

    # --- Main UI ---
    st.markdown(
        """
        <div class="fixed-header">
            <h1 style='font-size: 30px; color: #29B5E8; margin-bottom: 4px;'>
                Cortex AI – Property Management Insights by DiLytics
            </h1>
            <p style='font-size: 18px; color: #333;'>
                <strong>Welcome to Cortex AI. I am here to help with DiLytics Property Management Insights Solutions.</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    semantic_model_filename = SEMANTIC_MODEL.split("/")[-1]
    st.markdown(f"Semantic Model: `{semantic_model_filename}`")

    if st.session_state.show_greeting and not st.session_state.chat_history:
        st.markdown("Welcome! I’m the Snowflake AI Assistant, ready to assist you with property management. Ask about your rent, properties, leases, occupancy, or submit a maintenance request to get started!")
    else:
        st.session_state.show_greeting = False

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if message["role"] == "assistant" and "results" in message and message["results"] is not None:
                with st.expander("View SQL Query", expanded=False):
                    st.code(message["sql"], language="sql")
                st.markdown(f"**Query Results ({len(message['results'])} rows):**")
                st.dataframe(message["results"])
                if not message["results"].empty and len(message["results"].columns) >= 2:
                    st.markdown("**📈 Visualization:**")
                    display_chart_tab(message["results"], prefix=f"chart_{hash(message['content'])}", query=message.get("query", ""))

    chat_input_query = st.chat_input("Ask your question...")
    if chat_input_query:
        st.session_state.query = chat_input_query

    if st.session_state.query:
        query = st.session_state.query
        if query.lower().startswith("no of"):
            query = query.replace("no of", "number of", 1)
        st.session_state.show_greeting = False
        st.session_state.chart_x_axis = None
        st.session_state.chart_y_axis = None
        st.session_state.chart_type = "Bar Chart"
        original_query = query
        if query.strip().isdigit() and st.session_state.last_suggestions:
            try:
                index = int(query.strip()) - 1
                if 0 <= index < len(st.session_state.last_suggestions):
                    query = st.session_state.last_suggestions[index]
                else:
                    query = original_query
            except ValueError:
                query = original_query
        is_follow_up = any(re.search(pattern, query.lower()) for pattern in [r'^\bby\b\s+\w+$', r'^\bgroup by\b\s+\w+$']) and st.session_state.previous_query
        combined_query = query
        if st.session_state.use_chat_history and is_follow_up:
            chat_history = get_chat_history()
            if chat_history:
                combined_query = make_chat_history_summary(chat_history, query)
        st.session_state.chat_history.append({"role": "user", "content": original_query})
        st.session_state.messages.append({"role": "user", "content": original_query})
        with st.chat_message("user"):
            st.markdown(original_query)
        with st.chat_message("assistant"):
            with st.spinner("Generating Response..."):
                response_placeholder = st.empty()
                is_structured = is_structured_query(combined_query)
                is_complete = is_complete_query(combined_query)
                is_summarize = is_summarize_query(combined_query)
                is_suggestion = is_question_suggestion_query(combined_query)
                is_greeting = is_greeting_query(combined_query)
                assistant_response = {"role": "assistant", "content": "", "query": combined_query}
                response_content = ""
                failed_response = False

                if is_greeting or is_suggestion:
                    response_content = (
                        "Hello! Welcome to the Property Management AI Assistant!\n"
                        "Here are some questions you can try:\n"
                    )
                    suggestions = [
                        "Total number of properties currently occupied?",
                        "What is the average rent collected per tenant?",
                        "Total number of properties?",
                        "What’s the total rental income by property?"
                    ]
                    for i, suggestion in enumerate(suggestions, 1):
                        response_content += f"{i}. {suggestion}\n"
                    with response_placeholder:
                        for chunk in stream_text(response_content):
                            response_placeholder.markdown(response_content[:response_content.index(chunk) + len(chunk)], unsafe_allow_html=True)
                    assistant_response["content"] = response_content
                    st.session_state.last_suggestions = suggestions
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                elif is_complete:
                    response = create_prompt(combined_query)
                    if response:
                        response_content = f"**✍️ Generated Response:**\n{response}"
                        with response_placeholder:
                            for chunk in stream_text(response_content):
                                response_placeholder.markdown(response_content[:response_content.index(chunk) + len(chunk)], unsafe_allow_html=True)
                        assistant_response["content"] = response_content
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    else:
                        failed_response = True

                elif is_summarize:
                    summary = summarize(combined_query)
                    if summary:
                        response_content = f"**Summary:**\n{summary}"
                        with response_placeholder:
                            for chunk in stream_text(response_content):
                                response_placeholder.markdown(response_content[:response_content.index(chunk) + len(chunk)], unsafe_allow_html=True)
                        assistant_response["content"] = response_content
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    else:
                        failed_response = True

                elif is_structured:
                    response = snowflake_api_call(combined_query, is_structured=True)
                    sql, _ = process_sse_response(response, is_structured=True)
                    if sql:
                        results = run_snowflake_query(sql)
                        if results is not None and not results.empty:
                            results_text = results.to_string(index=False)
                            prompt = f"Provide a concise natural language answer to the query '{combined_query}' using the following data:\n\n{results_text}"
                            summary = complete(st.session_state.model_name, prompt)
                            if not summary:
                                summary = "Unable to generate a summary."
                            response_content = f"**✍️ Generated Response:**\n{summary}"
                            with response_placeholder:
                                for chunk in stream_text(response_content):
                                    response_placeholder.markdown(response_content[:response_content.index(chunk) + len(chunk)], unsafe_allow_html=True)
                            with st.expander("View SQL Query", expanded=False):
                                st.code(sql, language="sql")
                            st.markdown(f"**Query Results ({len(results)} rows):**")
                            st.dataframe(results)
                            if len(results.columns) >= 2:
                                st.markdown("**📈 Visualization:**")
                                display_chart_tab(results, prefix=f"chart_{hash(combined_query)}", query=combined_query)
                            assistant_response.update({
                                "content": response_content,
                                "sql": sql,
                                "results": results,
                                "summary": summary
                            })
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_content,
                                "sql": sql,
                                "results": results,
                                "summary": summary
                            })
                        else:
                            response_content = "No data returned for the query."
                            failed_response = True
                            assistant_response["content"] = response_content
                    else:
                        response_content = "Failed to generate SQL query."
                        failed_response = True
                        assistant_response["content"] = response_content

                else:
                    response = snowflake_api_call(combined_query, is_structured=False)
                    _, search_results = process_sse_response(response, is_structured=False)
                    if search_results:
                        raw_result = search_results[0]
                        response_content = f"**🔍 Generated Response:**\n{summarize_unstructured_answer(raw_result)}"
                        with response_placeholder:
                            for chunk in stream_text(response_content):
                                response_placeholder.markdown(response_content[:response_content.index(chunk) + len(chunk)], unsafe_allow_html=True)
                        assistant_response["content"] = response_content
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    else:
                        failed_response = True

                if failed_response:
                    suggestions = suggest_sample_questions(combined_query)
                    response_content = "I am not sure about your question. Here are some suggestions you can try:\n\n"
                    for i, suggestion in enumerate(suggestions, 1):
                        response_content += f"{i}. {suggestion}\n"
                    with response_placeholder:
                        for chunk in stream_text(response_content):
                            response_placeholder.markdown(response_content[:response_content.index(chunk) + len(chunk)], unsafe_allow_html=True)
                    assistant_response["content"] = response_content
                    st.session_state.last_suggestions = suggestions
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                st.session_state.chat_history.append(assistant_response)
                st.session_state.current_query = None
                st.session_state.current_results = assistant_response.get("results")
                st.session_state.current_sql = assistant_response.get("sql")
                st.session_state.current_summary = assistant_response.get("summary")
                st.session_state.previous_query = combined_query
                st.session_state.previous_sql = assistant_response.get("sql")
                st.session_state.previous_results = assistant_response.get("results")
                st.session_state.query = None
