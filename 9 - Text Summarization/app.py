import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Creating Streamlit app
st.set_page_config(page_title="Surya's Langchain: Summarize Text from YT and Website")
st.title("Surya's Langchain: Summarize Text from YT and Website")
st.subheader("It Summarizes your topic")

# Sidebar - Groq API Key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Input URL
generic_url = st.text_input("Enter URL Here")

# Validate API Key
if not groq_api_key.strip():
    st.error("Please provide a Groq API Key.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization Button
if st.button("Summarize the Content"):
    if not generic_url.strip():
        st.error("Please provide the URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL! It must be a YouTube or website URL.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # Load Content (YouTube or Website)
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(
                            generic_url, add_video_info=False
                        )
                        data = loader.load()
                    except Exception as yt_error:
                        st.error(f"Failed to fetch YouTube transcript: {yt_error}")
                        data = []
                else:
                    try:
                        loader = UnstructuredURLLoader(urls=[generic_url])
                        data = loader.load()
                    except Exception as web_error:
                        st.error(f"Failed to fetch website content: {web_error}")
                        data = []

                # Ensure valid content is extracted
                if not data or not isinstance(data, list):
                    st.error("Failed to retrieve content. Please check the URL and try again.")
                else:
                    # Summarization Chain (Passing Correct Input)
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run({"input_documents": data})

                    st.success("Summary Generated Successfully!")
                    st.write(output_summary)

        except Exception as e:
            st.error(f"An error occurred: {e}")
