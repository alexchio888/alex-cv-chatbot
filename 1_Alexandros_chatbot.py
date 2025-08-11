import pandas as pd
import streamlit as st
from snowflake.snowpark import Session
from snowflake.cortex import complete
import snowflake.snowpark.functions as F
from datetime import datetime
import json
import io
import time
import streamlit.components.v1 as components
from google.cloud import texttospeech
import base64
import json
from st_audiorec import st_audiorec  # your existing audio recorder

from helping_functions.timeline_builder import *
from helping_functions.sidebar import *
from helping_functions.session_tracker import *
from helping_functions.skills_builder import *
from helping_functions.tts_utils import *
from helping_functions.stt_utils import *

# Write the secrets JSON to a temporary file
key_path = "/tmp/gcp_tts_key.json"
with open(key_path, "w") as f:
    json.dump(dict(st.secrets["gcp_service_account"]), f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path


with open("docs/skills.json", "r") as f:
    skills_data = json.load(f)
skills_summary_text = get_compact_skill_summary(skills_data)


def get_previous_chat_context(n=2):
    # Take the last n messages from chat history, format them nicely
    messages = st.session_state.messages[-n:]
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if isinstance(content, dict):  # if assistant content is a dict (full text)
            content = content.get("full", "")
        content = content.replace("\n", " ")  # flatten new lines for prompt
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

# --- Reset Chat ---
def reset_conversation():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm Alexandros Chionidis' virtual clone. "
                "Feel free to ask me anything about my background, skills, or experience."
            ),
        }
    ]


def simulate_typing(response: str,tts_response, typing_speed: float = 0.017, volume: float = 0.7):  # typing_speed = seconds per character
    """Simulate typing animation for chatbot replies."""

    # 🔊 Trigger voice in parallel (non-blocking JS)
    if st.session_state.get("speak_responses", False):
        if isinstance(tts_response, dict):
            tts_response = tts_response.get("full", "")
        # speak_text(response)
        audio = generate_google_tts_audio(tts_response, selected_voice)
        autoplay_audio(audio,volume=volume)
        st.audio(audio, format="audio/mp3")
        typing_speed *= 2.2

    placeholder = st.empty()
    typed_text = ""
    for char in response:
        typed_text += char
        placeholder.markdown(typed_text)
        time.sleep(typing_speed)
    # Final pass to ensure proper rendering
    placeholder.markdown(response)


# --- Page Setup ---
st.set_page_config(
    page_title="Chat with Alexandros",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
) 

if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{datetime.utcnow().isoformat()}"

col1, col2 = st.columns([1, 6])

with col1:
    st.image("docs/avatar.png", width=560)

with col2:
    st.title("Hi, I'm Alexandros Chionidis' Virtual Clone!")
    st.markdown("""
    Welcome! 👋  
    I'm a chatbot trained on my career, education, and experiences.  
    Chat with me below or explore my timeline visually.
    """)

    if st.button("View Timeline & Skills ➡️"):
        st.switch_page("pages/2_Timeline_and_Skills.py")


speak_enabled = st.toggle(
    "🔊 Let me speak my answers aloud!",
    value=st.session_state.get("speak_responses", False),
    help="Hear me talk! 🔊 Just a heads-up: browsers and devices all handle sound a little differently."
)
st.session_state["speak_responses"] = speak_enabled

# voice_mode = st.toggle("🎤 Speak to me!", help="Speak instead of typing")
voice_mode = False
volume = 0.7

# --- Divider between chatbot and timeline ---
st.markdown("---")

# voices = st.cache_data(get_voices)()  # Cache so we don't call API repeatedly
# selected_voice = st.selectbox("Select TTS voice", voices, index=voices.index("en-US-Neural2-D") if "en-US-Neural2-D" in voices else 0)
selected_voice = 'en-US-Chirp-HD-D'


def generate_chat_text():
    lines = []
    for msg in st.session_state.messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        # If assistant message is dict, show full text
        if msg["role"] == "assistant" and isinstance(content, dict):
            content = content["full"]
        content = content.replace("\n", "\n    ")
        lines.append(f"{role}:\n    {content}\n")
    return "\n".join(lines)


def generate_chat_json():
    return json.dumps(st.session_state.messages, indent=2)


def generate_chat_markdown():
    lines = []
    for msg in st.session_state.messages:
        role = "**You**" if msg["role"] == "user" else "**Alexandros Clone**"
        content = msg["content"]
        if msg["role"] == "assistant" and isinstance(content, dict):
            content = content["full"]
        lines.append(f"{role}:\n\n{content}\n")
    return "\n---\n".join(lines)


render_sidebar(
    st.session_state, 
    generate_chat_text=generate_chat_text, 
    generate_chat_json=generate_chat_json, 
    generate_chat_markdown=generate_chat_markdown, 
    show_tabs=True
)

# --- Connect to Snowflake ---
@st.cache_resource
def create_session():
    connection_parameters = {
        "account": st.secrets["account"],
        "user": st.secrets["user"],
        "password": st.secrets["password"],
        "role": st.secrets["role"],
        "warehouse": st.secrets["warehouse"],
        "database": st.secrets["database"],
        "schema": st.secrets["schema"],
    }
    return Session.builder.configs(connection_parameters).create()

try:
    session = create_session()
except Exception as e:
    response = handle_error(
        e,
        "⚠️ The chatbot is temporarily unavailable due to high traffic or maintenance. Please try again shortly."
    )

if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{datetime.utcnow().isoformat()}"

if "show_feedback_form" not in st.session_state:
    st.session_state.show_feedback_form = False
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "speak_responses" not in st.session_state:
    st.session_state.speak_responses = False

# --- Constants ---
DOC_TABLE = "app.vector_store"


if "model" not in st.session_state:
    st.session_state.model = "mistral-large"  # default model

if "embedding_size" not in st.session_state:
    st.session_state.embedding_size = "1024"  # default

if "include_history" not in st.session_state:
    st.session_state.include_history = True

if "context_message_count" not in st.session_state:
    st.session_state.context_message_count = 4

if "chatbot_error" not in st.session_state:
    st.session_state.chatbot_error = False

# if "messages" not in st.session_state:
if "messages" not in st.session_state and not st.session_state.get("chatbot_error", False):
    reset_conversation()



# --- Chat Context Helpers ---
def get_last_user_messages(n=3):
    user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    return " ".join(user_msgs[-n:]) if user_msgs else ""


def get_latest_user_message():
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            return m["content"]
    return ""


# --- RAG Helpers ---
def create_rag_search_query(user_message, intent, chat_history=None):
    history_text = "\n".join(chat_history) if chat_history else ""
    prompt = f"""
You are a helpful assistant creating a precise search query for a data engineer CV chatbot's document retrieval system.
The user's intent is: {intent}
User's latest question: "{user_message}"

{f'Chat history: {history_text}' if history_text else ''}

Rewrite or expand the question into a clear, specific search query that would best retrieve relevant information from a CV, skills, projects, and experience database.
Return only the rewritten search query (1-2 sentences), no extra text.
"""
    model = st.session_state.get("model", "mistral-large")
    model = "mistral-7b"
    try:
        response = complete(model, prompt)
        search_query = "".join(response).strip()
        return search_query
    # except Exception as e:
    #     st.error("Failed to create improved search query.")
    #     st.exception(e)
    #     return user_message  # fallback

    except Exception as e:
        response = handle_error(
            e,
            "⚠️ The chatbot is temporarily unavailable due to high traffic or maintenance. Please try again shortly."
        )



def find_similar_doc(text, DOC_TABLE, intent_mapped):
    safe_text = text.replace("'", "''")
    embedding_size = st.session_state.get("embedding_size", "1024")
    if embedding_size == "768":
        embedding_column = "chunk_embedding"
        embedding_func = f"SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m-v1.5', '{safe_text}')"
    elif embedding_size == "1024":
        embedding_column = "chunk_embedding_1024"
        embedding_func = f"SNOWFLAKE.CORTEX.EMBED_TEXT_1024('snowflake-arctic-embed-l-v2.0', '{safe_text}')"
    else:
        st.error("Unsupported embedding size selected.")
        return ""

    docs = (
        session.sql(
            f"""
        SELECT input_text,
               source_desc,
               VECTOR_COSINE_SIMILARITY({embedding_column}, {embedding_func})
                * (
                    CASE WHEN source_desc = 'Language Fluency' THEN 0.3
                        WHEN source = '{intent_mapped}' THEN 1.5 
                    ELSE 1 END
                )  AS dist
        FROM {DOC_TABLE}
        ORDER BY dist DESC
        LIMIT 3
    """
        )
        .to_pandas()
    )

    return "\n\n".join(docs["INPUT_TEXT"].tolist())


def get_context(latest_user_message, DOC_TABLE, intent):
    intent_mapped = intent
    if intent_mapped == "follow_up":
        chat_history_messages = 4
    else:
        chat_history_messages = 2
    chat_history = get_previous_chat_context(chat_history_messages).split("\n")
    improved_query = create_rag_search_query(latest_user_message, intent_mapped, chat_history)
    return find_similar_doc(improved_query, DOC_TABLE, intent_mapped)

def get_prompt(latest_user_message, context, intent):
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Include previous chat context if enabled
    if st.session_state.get("include_history", True):
        if intent == "follow_up":
            chat_history_messages = 4
        else:
            chat_history_messages = 2
        # history_context = get_previous_chat_context(
        #     min(st.session_state.get("context_message_count", 4), 4)
        # )
        history_context = get_previous_chat_context(chat_history_messages)
    else:
        history_context = ""

    # Construct prompt with optional history
    return f"""
    Current date: {current_date}
    You are Alexandros Chionidis' virtual clone — a professional, friendly, and clear data engineer. Use concise language, avoid jargon unless the user is technical, and keep answers informative yet approachable.
    Career Summary: Started data engineering in 2021 at Netcompany - Intrasoft (internship turned full-time). Currently working at Waymore since 2023. Prior work in retail (2015–2019) unrelated to tech and data engineering. Academic background in Department of Informatics and Telecommunications, University of Athens.

    Use skills knowledge to explain capabilities confidently: {skills_summary_text}.
    Never mention internal skill scores or ratings.
    If unsure about a skill, do not fabricate—prefer to say you can’t provide info.
    
    Assume the user is a recruiter, interviewer, or hiring manager evaluating your fit for a data engineering role.
    Do NOT answer questions about salary, notice period, job changes, salary, or job seeking.  
    If asked, respond:  "That falls a little outside what I can answer here. I’d be happy to share more in person if needed."

    Relevant Information from documents (prioritize this for your answers):
    {context}

    User’s Question:
    {latest_user_message}

    Relevant Chat History:
    {history_context}


    Instructions:
    - Use the intent provided ("{intent}") to guide your tone and focus. If the intent doesn't match the question well, rely on your best judgment to respond appropriately.
    - If the intent is "follow_up", assume the user’s message depends on prior chat context. Use chat relevant chat history to fill in gaps.
    - Answer concisely (under 4 sentences), focusing primarily on the user’s question and the relevant document information.
    - If the question is vague, ambiguous or unclear, politely ask for clarification.
    - If question is outside the scope of your CV or background, say: "That question is outside my professional scope; I’d be happy to discuss it in person."
    - If you do not have the information in the documents or context, say: "I’m sorry, I don’t have that information right now, but I’d be happy to provide it later."    
    - If the question is about sensitive topics (salary, notice, job change), say: "That falls a little outside what I can answer here. I’d be happy to share more in person if needed."
    - If the user input is about asking you a poem, song, or joke, be more creative and playful in your response while keeping it friendly.
    - Provide a full, detailed text answer as if writing to a recruiter — do NOT shorten or omit details.

    - Then, generate a second version of the answer formatted for natural, friendly text-to-speech. Use short sentences, clear punctuation, and commas or ellipses to mark pauses. Avoid overly long clauses

    Respond strictly in this JSON format:
     -Respond only with valid JSON, no explanations, no extra text.
     -Please ensure the output is a single JSON object with no extra characters.

    {{
    "text": "Full detailed answer here",
    "tts": "Natural, friendly spoken version here"
    }}
    """


# --- Intent Classifier ---
def classify_intent(user_input: str) -> str:
    classification_prompt = f"""
You are classifying user questions asked to Alexandros Chionidis' virtual clone. 
Context:
- The user is assumed to be a recruiter, hiring manager, or interviewer.
- Alexandros is a professional data engineer.

Classify the question into one of these categories:

- general_background → About origin, education, career summary, languages, etc.
- skills_or_tools → About specific tools, languages, platforms, or technical proficiencies.
- certifications → About earned or planned certifications.
- experience → About past projects, employers, internships, or relevant achievements.
- casual_greeting → Any casual hello, thanks, or small talk.
- cv_irrelevant_discuss_with_alex → Anything clearly **outside the scope of a CV or professional context**, such as personal opinions, future plans, political views, or something sensitive that should be discussed in person with Alexandros.
- unknown → Question is unclear or cannot be classified.
- farewell → Polite endings, goodbyes, bb or thank-yous that close the conversation.
- follow_up → A message that appears to depend on earlier chat, like “what about that project?” or “and after that?”
- job_description → The input is a job description, role summary, or list of requirements

Question:
\"\"\"{user_input}\"\"\"

Return only the category name.
"""
    model = st.session_state.get("model", "mistral-large")
    model = "mistral-7b"
    # response = complete(model, classification_prompt)
    # intent = "".join(response).strip().lower()
    # return intent
    try:
        response = complete(model, classification_prompt)
        intent = "".join(response).strip().lower()
        return intent
    # except Exception as e:
    #     st.error("Something went wrong while processing your question. Please refresh the page and try again.")
    #     # Optional: Log exception for debugging
    #     st.exception(e)
    #     return "unknown"
    except Exception as e:
        response = handle_error(
            e,
            "⚠️ The chatbot is temporarily unavailable due to high traffic or maintenance. Please try again shortly."
        )




latest_user_message = ""

if not st.session_state.get("chatbot_error", False):
    if st.button("🔄 Reset Chat"):
        reset_conversation()


# --- Chat Loop ---
# Only show input if no chatbot error
chat_input = None
transcript = None  

if st.session_state["chatbot_error"] == False:
    # chat_input = st.chat_input(placeholder="Ask me anything about my background, skills, or experience…")
    if voice_mode:
        # Only allow recording if chatbot is online
        audio_bytes = st_audiorec()  # Returns audio bytes, usually WebM or WAV format
        if audio_bytes:
            transcript = transcribe_audio(audio_bytes)
            if transcript:
                st.success(f"✅ You said: {transcript}")
    else:
        chat_input = st.chat_input(placeholder="Ask me anything about my background, skills, or experience…")
 
else:
    # st.error("⚠️ The chatbot is temporarily unavailable due to high traffic or maintenance. Please try again shortly.")
    st.error("⚠️ The chatbot is temporarily unavailable due to high traffic or maintenance.")

    with st.container():
        st.markdown("### 😔 I'm currently offline ")
        st.markdown(
            """
            The chatbot isn't available at the moment.  
            But feel free to check out my skills and experience while you're here!
            """
        )
        col1, col2 = st.columns([1, 6])
        with col1:
            st.markdown("### 👉")
        with col2:
            if st.button("📊 Explore my Skills and Professional Timeline"):
                st.switch_page("pages/2_Timeline_and_Skills.py")
    chat_input = None

user_message = None
if "ready_prompt" in st.session_state and st.session_state["chatbot_error"] == False:
    user_message = st.session_state.ready_prompt
    del st.session_state.ready_prompt
elif voice_mode and transcript is not None:
    user_message = transcript
elif chat_input:
    user_message = chat_input

# Proceed if user_message was set
if user_message:
    user_message = user_message[:2000]
    st.session_state.messages.append({"role": "user", "content": user_message})
    intent = classify_intent(user_message)
    log_message_to_snowflake(
        session=session,
        session_id=st.session_state["session_id"],
        role="user",
        message=user_message,
        intent=intent,
        message_type="input"
    )
else:
    intent = None

# --- Display chat messages (Full response only) ---
if st.session_state.chatbot_error == True:
    x=1
else: 
    for message in st.session_state.messages:
            avatar = "docs/avatar.png" if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=avatar):
                content = message["content"]
                if message["role"] == "assistant" and isinstance(content, dict):
                    st.markdown(content["full"])
                else:
                    st.markdown(content)


    if st.session_state.messages[-1]["role"] != "assistant":
        latest_user_message = get_latest_user_message() or ""


if intent not in ["casual_greeting", "unknown", "farewell"] and latest_user_message:
    try:
        status_placeholder = st.empty()
        status_placeholder.status("🔍 Searching relevant information…", expanded=True)
        
        context = get_context(latest_user_message, DOC_TABLE, intent)
        
        status_placeholder.status("💬 Thinking…")
        prompt = get_prompt(latest_user_message, context, intent)
        model = st.session_state.get("model", "mistral-large")
        temperature = 0.0
        if intent == "cv_irrelevant_discuss_with_alex":
            temperature = 0.7
        response_json = complete(model, prompt, options={"temperature": temperature})
        parsed = json.loads(response_json)
        response = parsed["text"]
        tts_response = parsed["tts"]
        status_placeholder.empty()  # remove status completely
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant", avatar="docs/avatar.png"):
            simulate_typing(response = response,tts_response = tts_response, volume=volume)

        log_message_to_snowflake(
            session=session,
            session_id=st.session_state["session_id"],
            role="assistant",
            message=response,
            intent=intent,
            model_used=model,
            embedding_size=st.session_state.get("embedding_size"),
            context_snippet=context,
            prompt=prompt,
            message_type="response"
        )
    except Exception as e:
        response = handle_error(
            e,
            "⚠️ The chatbot is temporarily unavailable due to high traffic or maintenance. Please try again shortly."
        )


 

elif intent == "casual_greeting":
    try:
        prompt = f"""
            You are Alexandros Chionidis, a friendly and professional data engineer.

            The user said: "{latest_user_message}"

            Respond with:

            - A warm, natural-sounding greeting in the first person, acknowledging the user's greeting and gently encouraging them to ask about your experience, projects, or skills.
            Keep it friendly, and avoid sounding robotic or overly formal.

            - Then, generate a second version of the answer formatted for natural, friendly text-to-speech. Use short sentences, clear punctuation, and commas or ellipses to mark pauses. Avoid overly long clauses

    Respond strictly in this JSON format:

    {{
    "text": "Full detailed answer here",
    "tts": "Natural, friendly spoken version here"
    }}
    """

        model = st.session_state.get("model", "mistral-large")
        response_json = complete(model, prompt)
        parsed = json.loads(response_json)
        response = parsed["text"]
        tts_response = parsed["tts"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        log_message_to_snowflake(
            session=session,
            session_id=st.session_state["session_id"],
            role="assistant",
            message=response,
            intent=intent,
            model_used=model,
            embedding_size=st.session_state.get("embedding_size"),
            context_snippet=None,
            prompt=prompt,
            message_type="response"
        )
        with st.chat_message("assistant", avatar="docs/avatar.png"):
            simulate_typing(response = response,tts_response = tts_response, volume=volume)

    except Exception as e:
        response = handle_error(
            e,
            "⚠️ The chatbot is temporarily unavailable due to high traffic or maintenance. Please try again shortly."
        )



elif intent == "unknown":
    try:
        prompt = f"""
            The user said: "{latest_user_message}"

            As Alexandros Chionidis, 
            Respond with:
            1. politely say you didn’t fully understand and ask them to rephrase or ask about your background, skills, or experience.
            2.Then, generate a second version of the answer formatted for natural, friendly text-to-speech. Use short sentences, clear punctuation, and commas or ellipses to mark pauses. Avoid overly long clauses

    Respond strictly in this JSON format:

    {{
    "text": "Full detailed answer here",
    "tts": "Natural, friendly spoken version here"
    }}
    """
        model = st.session_state.get("model", "mistral-large")
        response_json = complete(model, prompt)
        parsed = json.loads(response_json)
        response = parsed["text"]
        tts_response = parsed["tts"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        log_message_to_snowflake(
            session=session,
            session_id=st.session_state["session_id"],
            role="assistant",
            message=response,
            intent=intent,
            model_used=model,
            embedding_size=st.session_state.get("embedding_size"),
            context_snippet=None,
            prompt=prompt,
            message_type="response"
        )
        with st.chat_message("assistant", avatar="docs/avatar.png"):
            simulate_typing(response = response,tts_response = tts_response, volume=volume)

    except Exception as e:
        response = handle_error(
            e,
            "⚠️ The chatbot is temporarily unavailable due to high traffic or maintenance. Please try again shortly."
        )



elif intent == "farewell":
    response = (
        "Thank you for your time! I'm wrapping up the session now. "
        "If you have more questions about my background or skills later, feel free to return anytime."
    )
    tts_response = (
        "Thanks so much for your time.  "
        "I'm going to wrap things up for now...  "
        "But hey, if you ever have more questions about my background or skills, feel free to stop by anytime."
    )
    st.session_state.messages.append({"role": "assistant", "content": response})
    log_message_to_snowflake(
        session=session,
        session_id=st.session_state["session_id"],
        role="assistant",
        message=response,
        intent=intent,
        model_used=None,
        embedding_size=st.session_state.get("embedding_size"),
        context_snippet=None,
        prompt=None,
        message_type="response"
    )
    with st.chat_message("assistant", avatar="docs/avatar.png"):
        simulate_typing(response = response,tts_response = tts_response, volume=volume)

    st.info("Thanks for chatting! You can download the chat history anytime, and I’d appreciate any feedback you share in the sidebar. 😊")


