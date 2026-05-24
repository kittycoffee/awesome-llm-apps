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
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1" # <--- 注意这里！把 API_BASE 改成了 BASE_URL

# 【🌟 新增这一行：解决 HuggingFace 下载模型 SSL 断连崩溃问题】
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Set up the Streamlit App
st.title("AI Travel Agent with Memory 🧳")
st.caption("基于 DeepSeek 和云端 Qdrant 的智能记忆旅行代理。")

# ==========================================
# 核心魔改 2：直接在后台初始化客户端，跳过网页输入，开发更顺畅
# ==========================================
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    # 最新版的 OpenAI 官方库中，用来修改网址的环境变量已经从以前的 OPENAI_API_BASE 悄悄改成了 OPENAI_BASE_URL
    base_url=os.environ["OPENAI_BASE_URL"] # <--- 这里也要跟着改成 BASE_URL
)

# 专门为旅行助理定制的中文记忆提取 Prompt
chinese_prompt = """
你是一个专业的中文旅行记忆提取专家。
请从用户的输入中，提取出关于用户的关键客观事实（如：身份、饮食偏好、过敏史、喜欢的景点类型、预算限制等）。
【严格禁止】：不要记录任何关于行程推荐、景点介绍或导游回复的内容。
【语言要求】：请务必使用**中文**进行总结，语言要简练客观，例如“用户不喜欢吃香菜”。
如果没有值得记忆的个人信息，请直接返回空。
"""

# 终极混合架构：云端向量库 + DeepSeek 聊天 + 本地化向量提取
config = {
    # 【新增这一行】：强行覆盖它的默认英文系统提示词
    "custom_prompt": chinese_prompt,

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
            "temperature": 0.1,
    
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
st.sidebar.title("请输入你的用户名:")
previous_user_id = st.session_state.get("previous_user_id", None)
user_id = st.sidebar.text_input("用户名")

# 切换用户时清空当前屏幕的聊天记录
if user_id != previous_user_id:
    st.session_state.messages = []
    st.session_state.previous_user_id = user_id

# --- 侧边栏：查看记忆按钮 ---
st.sidebar.title("记忆点")
# 1. 把按钮放到侧边栏
if st.sidebar.button("查看我的记忆点"):
    # 2. 只有点击了按钮，才去判断有没有填用户名
    if user_id: 
        memories = memory.get_all(filters={"user_id": user_id})
        # 3. 安全提取记忆结果
        if memories and "results" in memories and len(memories["results"]) > 0:
            st.sidebar.write(f"关于**{user_id}**的历史记忆:")
            for mem in memories["results"]:
                if "memory" in mem:
                    # 用 success 的绿色框展示记忆，视觉效果更好
                    st.sidebar.success(f"- {mem['memory']}")
        else:
            st.sidebar.info("当前用户还没有任何记忆数据。快去聊点什么吧！")
    else:
        # 如果没填用户名就瞎点按钮，才给红色报错
        st.sidebar.error("请先在上方输入用户名！")

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("世界很大，你想去哪里旅游呢？")  # <--- 把提示语改成更有旅行氛围的版本

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

   # ==========================================
    # 核心优化：只对用户的输入进行记忆提炼，并强行注入中文 Prompt
    # ==========================================
    user_mem = memory.add(
        prompt,  # 只喂给它用户说的话，不要喂 answer！
        user_id=user_id, 
        metadata={"role": "user"},
        prompt=chinese_prompt  # <--- 秘密武器：在这里把中文指令传给它！
    )
    print("【后台提炼的用户记忆】:", user_mem)

elif not user_id:
    st.error("⚠️ 请先在左侧侧边栏输入您的用户名，然后才能开始聊天哦！")
