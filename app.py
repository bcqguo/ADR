# app.py

#https://github.com/ossirytk/llama-cpp-langchain-chat
#https://docs.streamlit.io/library/api-reference/text/st.header

from typing import List, Union
import pickle, os, sys
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
from common import set_page_container_style
from io import StringIO

#set1, set2 = pickle.load(open('train_set_sample_lst.pkl', "rb"))


def init_page() -> None:    
    
    st.set_page_config(
        page_title="ADR-GPT"
    )
    #st.sidebar.image('./utsa_logo.png')
    
    set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 0, padding_left = 0, padding_bottom = 0)
           
    st.sidebar.header(":blue[ADR-GPT: Chatbot for ADR]", divider='rainbow')
    st.sidebar.write("ADR-GTP is built on top of a large language model trained on quarterly data from FDA Adverse Event Reporting System (FAERS) between 2012 and 2023, using pre-trained Llama_2_7b_chat_hf as the base model.")    
    # st.markdown("""
    #     <style>
    #            .block-container {
    #                 padding-top: -3rem;
    #                 padding-bottom: 0rem;
    #                 padding-left: 0rem;
    #                 padding-right: 0rem;
    #             }
    #     </style>
    #     """, unsafe_allow_html=True)
    
def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="AI assistant for ADR will reply your query in mardkown format.")
        ]
        st.session_state.costs = []


def select_llm() -> Union[ChatOpenAI, LlamaCpp]:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("ADR_Llama_2_7b_chat_hf_0",
                                   "ADR_Llama_2_7b_chat_hf_1",
                                   "ADR_Llama_2_7b_chat_hf_2",
                                   "ADR_Llama_2_7b_chat_hf_merged",
                                   "Llama-2-7b-chat-hf",
                                   ))
    temperature = st.sidebar.slider("Temperature (Creativity):", min_value=0.0,
                                    max_value=1.0, value=0.2, step=0.01)
    
    top_p = st.sidebar.slider("top_p (Token Randomness):", min_value=0.1,
                                    max_value=1.0, value=0.8, step=0.1)
    
    max_tokens = st.sidebar.slider("Maximal Tokens:", min_value=10,
                                    max_value=4096, value=512, step=1)
    
    if model_name == "Llama-2-7b-chat-hf":
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path= f"{model_name}.bin",
            max_tokens= max_tokens,
            temperature=temperature,
            stop=["</s>"], 
            top_p=top_p,
            #top_k=10,
            repeat_penalty=2,   
            callback_manager=callback_manager,
            verbose=True,  # False
        )
    
#     # elif model_name = "gpt-4":
    #     return ChatOpenAI(temperature=temperature, 
    #                       openai_api_key=openai_api_key,
    #                       model_name=model_name)
    
    elif model_name == "ADR_Llama_2_7b_chat_hf_0":
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path= '7b_adr_0_ggml.bin',
            max_tokens= max_tokens,
            temperature=temperature,
            stop=["</s>"], 
            top_p=top_p,
            #top_k=10,
            repeat_penalty=2,   
            callback_manager=callback_manager,
            verbose=True,  # False
        )
    elif model_name == "ADR_Llama_2_7b_chat_hf_1":
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path= '7b_adr_1_ggml.bin',
            max_tokens= max_tokens,
            temperature=temperature,
            stop=["</s>"], 
            top_p=top_p,
            #top_k=10,
            repeat_penalty=2,   
            callback_manager=callback_manager,
            verbose=True,  # False
        )
    
    elif model_name == "ADR_Llama_2_7b_chat_hf_2":
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path= '7b_adr_2_ggml.bin',
            max_tokens= max_tokens,
            temperature=temperature,
            stop=["</s>"], 
            top_p=top_p,
            #top_k=10,
            repeat_penalty=2,   
            callback_manager=callback_manager,
            verbose=True,  # False
        )
            
    elif model_name == "ADR_Llama_2_7b_chat_hf_merged":
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path= '7b_merge_ggml.bin',
            max_tokens= max_tokens,
            temperature=temperature,
            stop=["</s>"], 
            top_p=top_p,
            #top_k=10,
            repeat_penalty=2,   
            callback_manager=callback_manager,
            verbose=True,  # False
        )
          

def get_answer(llm, messages) -> tuple[str, float]:
    if isinstance(llm, ChatOpenAI):
        with get_openai_callback() as cb:
            answer = llm(messages)
        return answer.content, cb.total_cost
    if isinstance(llm, LlamaCpp):
        return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0


def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""84.0 YR F 81.72 KG NATEGLINIDE. Oral TAKE 15 MIN BEFORE MEALS WARFARIN Oral M/W/F 5 MG TAB ONCE DAILY; T/T/S/S 1/2 OF A 5 MG TAB ONCE DAILY Hypothyroidism Thyroid disorder Diabetes mellitus Anxiety Product used for unknown indication Product used for unknown indication Ulcer Neuropathy peripheral Neuropathy peripheral Neuropathy peripheral Product used for unknown indication Product used for unknown indication Vitamin supplementation Atrial fibrillation"""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list).replace('Product used for unknown indication ', '')


def main() -> None:
    _ = load_dotenv(find_dotenv())

    init_page()
    llm = select_llm()
    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Input your query"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ADR_GPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
    
    #Upload multipel files                
    uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)

    # costs = st.session_state.get("costs", [])
    # st.sidebar.markdown("## Costs")
    # st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    # for cost in costs:
    #     st.sidebar.markdown(f"- ${cost:.5f}")


# streamlit run app.py
if __name__ == "__main__":
    main()