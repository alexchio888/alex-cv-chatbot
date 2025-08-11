from snowflake.snowpark import Session
from datetime import datetime
import uuid
import streamlit as st
import streamlit.components.v1 as components
import time

TABLE_NAME = "CHAT_LOGS"
def reset_chat():
    keys_to_clear = ["messages", "chatbot_error", "error_shown", "ready_prompt", "session_id", "other_state_vars"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def log_message_to_snowflake(
    session: Session,
    session_id: str,
    role: str,
    message: str,
    *,
    user_id: str = None,
    intent: str = None,
    model_used: str = None,
    embedding_size: str = None,
    context_snippet: str = None,
    prompt: str = None,
    message_type: str = None
):
    timestamp = datetime.utcnow().isoformat()
    
    # Escape strings
    escape = lambda s: s.replace("'", "''") if s else None
    session_id = escape(session_id)
    role = escape(role)
    message = escape(message)[:5000] if message else None
    prompt = escape(prompt)[:64000] if prompt else None
    context_snippet = escape(context_snippet)[:64000] if context_snippet else None
    user_id = escape(user_id)

    session.sql(f"""
        INSERT INTO {TABLE_NAME} (
            session_id, user_id, timestamp, role, message,
            intent, model_used, embedding_size,
            context_snippet, prompt, message_type
        )
        VALUES (
            '{session_id}', {'NULL' if not user_id else f"'{user_id}'"}, '{timestamp}', '{role}', {f"'{message}'" if message else 'NULL'},
            {'NULL' if not intent else f"'{escape(intent)}'"},
            {'NULL' if not model_used else f"'{escape(model_used)}'"},
            {'NULL' if not embedding_size else f"'{escape(embedding_size)}'"},
            {'NULL' if not context_snippet else f"'{context_snippet}'"},
            {'NULL' if not prompt else f"'{prompt}'"},
            {'NULL' if not message_type else f"'{escape(message_type)}'"}
        )
    """).collect()



def ensure_user_id():
    """Injects JS to persist and communicate a user_id cookie."""
    if "user_id" in st.session_state:
        return

    generated_id = str(uuid.uuid4())

    components.html(f"""
        <script>
        const COOKIE_NAME = "user_id";

        function setCookie(name, value, days) {{
            let expires = "";
            if (days) {{
                const date = new Date();
                date.setTime(date.getTime() + (days*24*60*60*1000));
                expires = "; expires=" + date.toUTCString();
            }}
            document.cookie = name + "=" + (value || "") + expires + "; path=/";
        }}

        function getCookie(name) {{
            const nameEQ = name + "=";
            const ca = document.cookie.split(';');
            for (let i = 0; i < ca.length; i++) {{
                let c = ca[i].trim();
                if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
            }}
            return null;
        }}

        const uid = getCookie(COOKIE_NAME) || "{generated_id}";
        setCookie(COOKIE_NAME, uid, 365);

        // Send it to Streamlit
        const streamlitEvent = new CustomEvent("streamlit:user_id", {{
            detail: uid
        }});
        window.dispatchEvent(streamlitEvent);
        </script>
    """, height=0)

def handle_error(e: Exception, user_friendly_message: str = "An unexpected error occurred."):
    """
    General-purpose error handler.
    
    Args:
        e (Exception): The exception to handle.
        user_friendly_message (str): Optional message to show users.
    """
    reset_chat()
    # Optionally display a user-facing error
    st.error(f"❌ {user_friendly_message}")

    # Expandable for developers
    with st.expander("🔍 See technical details"):
        st.exception(e)

    # Optional: Set maintenance flag in session state (or even secrets if needed)
    st.session_state["chatbot_error"] = True

    # Optional: You can trigger logging to Snowflake or another service here
    # log_exception_to_snowflake(e)  # example placeholder

    ##### st.rerun() WILL REMOVE THE ERRORS FROM SCREEN.
    st.rerun()
    # time.sleep(1000000)
    return "Something went wrong. Please try again later."