import streamlit as st
from openai import OpenAI
from mem0 import Memory
import os
from dotenv import load_dotenv # 导入 dotenv

# 加载 .env 文件里的环境变量
load_dotenv()
# ==========================================
# 核心魔改 1：全局环境变量劫持 (非常重要)
# 这样不仅是你自己的代码，连 Mem0 在后台都会被迫乖乖使用 DeepSeek
# ==========================================
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY") 
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# Set up the Streamlit App
st.title("AI Travel Agent with Memory 🧳")
st.caption("基于 DeepSeek 和云端 Qdrant 的智能记忆旅行代理。")

# ==========================================
# 核心魔改 2：直接在后台初始化客户端，跳过网页输入，开发更顺畅
# ==========================================
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"]
)


# 终极混合架构：云端向量库 + DeepSeek 聊天 + 本地化向量提取
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "travel_memories_384",  # <--- 【加上这一行！】给新仓库起个专属名字
            "host": "124.220.72.84",  # <--- 你的腾讯云公网 IP
            "port": 6333,
            "embedding_model_dims": 384  # <--- 【关键修复！】强行锁定数据库维度为 384
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "deepseek-chat", # 告诉 Mem0 用 DeepSeek 来提取记忆
            "temperature": 0.1
        }
    },
    # 【新增这一块】：强制指定 Mem0 使用 HuggingFace 的本地模型来计算向量
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }
}
memory = Memory.from_config(config)

# ==========================================
# 4. 前端 UI 与业务流转逻辑
# ==========================================

# --- 侧边栏：用户身份认证与记忆查看 ---
st.sidebar.title("Enter your username:")
previous_user_id = st.session_state.get("previous_user_id", None)
user_id = st.sidebar.text_input("Enter your Username")

# 切换用户时清空当前屏幕的聊天记录
if user_id != previous_user_id:
    st.session_state.messages = []
    st.session_state.previous_user_id = user_id

# Sidebar option to show memory
st.sidebar.title("Memory Info")
if st.button("View My Memory"):
    memories = memory.get_all(filters={"user_id": user_id})
    if memories and "results" in memories:
        st.write(f"Memory history for **{user_id}**:")
        for mem in memories["results"]:
            if "memory" in mem:
                st.write(f"- {mem['memory']}")
    else:
        st.sidebar.info("No learning history found for this user ID.")
else:
    st.sidebar.error("Please enter a username to view memory info.")

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Where would you like to travel?")

if prompt and user_id:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant memories
    # 新版的 mem0ai 里，搜索记忆时不能直接把 user_id 扔进去，而是必须把它包装在一个 filters（过滤器）字典里。
    relevant_memories = memory.search(query=prompt, filters={"user_id": user_id})
    context = "Relevant past information:\n"
    if relevant_memories and "results" in relevant_memories:
        for mem in relevant_memories["results"]:
            if "memory" in mem:
                context += f"- {mem['memory']}\n"

    # Prepare the full prompt
    full_prompt = f"{context}\nHuman: {prompt}\nAI:"

    # Generate response
    response = client.chat.completions.create(
        model="deepseek-chat",  # <--- 这里把 "gpt-4o" 改成 "deepseek-chat"
        messages=[
            {"role": "system", "content": "You are a travel assistant with access to past conversations."},
            {"role": "user", "content": full_prompt}
        ]
    )
    answer = response.choices[0].message.content

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Store the user query and AI response in memory
    memory.add(prompt, user_id=user_id, metadata={"role": "user"})
    memory.add(answer, user_id=user_id, metadata={"role": "assistant"})
elif not user_id:
    st.error("⚠️ 请先在左侧侧边栏输入您的用户名，然后才能开始聊天哦！")
