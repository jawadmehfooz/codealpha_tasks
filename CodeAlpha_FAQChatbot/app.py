"""Streamlit chat UI for the CodeAlpha FAQ Chatbot.

Run with:  streamlit run app.py
"""

import streamlit as st

from chatbot import FAQChatbot, load_faqs


@st.cache_resource
def get_bot() -> FAQChatbot:
    return FAQChatbot(load_faqs())


def main() -> None:
    st.set_page_config(page_title="CodeAlpha FAQ Chatbot", page_icon="🤖")
    st.title("🤖 CodeAlpha FAQ Chatbot")
    st.write(
        "Ask me about CodeAlpha, the AI internship, or general AI topics — "
        "for example, *What is machine learning?*"
    )

    bot = get_bot()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm the CodeAlpha FAQ chatbot. What would you like to know?",
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        answer = bot.get_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
