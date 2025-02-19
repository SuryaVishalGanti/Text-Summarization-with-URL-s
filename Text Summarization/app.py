import validators
import streamlit as st
import re
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader, YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.schema import Document  # ✅ Fix: Ensure correct input format

# Streamlit App Setup
st.set_page_config(page_title="Surya's Langchain: Summarize Text from YT and Website")
st.title("Surya's Langchain: Summarize Text from YT and Website")
st.subheader("Summarizes Videos & Articles in Any Language")

# Sidebar - API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Input URL
generic_url = st.text_input("Enter YouTube or Website URL")

# Validate API Key
if not groq_api_key.strip():
    st.error("Please provide a Groq API Key.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a detailed summary of the following content in 300 words, with Perfect Heading in English:
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Extract YouTube Video ID
def extract_youtube_id(url):
    """Extracts the video ID from a YouTube URL."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Fetch YouTube Transcript (Any Language)
def get_youtube_transcript(video_id):
    """Retrieves YouTube video transcript in English or another available language."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get an English transcript first, otherwise take the first available
        transcript = None
        for transcript_option in transcript_list:
            if transcript_option.language_code == "en":
                transcript = transcript_option
                break
        if transcript is None:
            transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
        
        return " ".join([entry["text"] for entry in transcript.fetch()])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

# Summarization Button
if st.button("Summarize the Content"):
    if not generic_url.strip():
        st.error("Please provide a valid URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL! Please enter a YouTube or website URL.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                text_content = ""

                # YouTube Video Processing
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = extract_youtube_id(generic_url)
                    
                    if not video_id:
                        st.error("Invalid YouTube URL format.")
                    else:
                        try:
                            # Attempt to get the transcript
                            text_content = get_youtube_transcript(video_id)

                            # If transcript is missing, use video metadata (title & description)
                            if not text_content:
                                loader = YoutubeLoader.from_youtube_url(
                                    generic_url, add_video_info=True
                                )
                                data = loader.load()
                                text_content = " ".join(
                                    [doc.page_content for doc in data if hasattr(doc, "page_content")]
                                )

                        except Exception as yt_error:
                            st.error(f"Error fetching YouTube content: {yt_error}")
                
                # Website Processing
                else:
                    try:
                        loader = UnstructuredURLLoader(urls=[generic_url])
                        data = loader.load()
                        text_content = " ".join(
                            [doc.page_content for doc in data if hasattr(doc, "page_content")]
                        )
                    except Exception as web_error:
                        st.error(f"Error fetching website content: {web_error}")

                # Ensure valid content is extracted
                if not text_content or not isinstance(text_content, str) or text_content.strip() == "":
                    st.error("Failed to retrieve content. Please check the URL and try again.")
                else:
                    # ✅ Fix: Convert extracted text into a `Document` object
                    document = [Document(page_content=text_content)]

                    # Summarization Chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run({"input_documents": document})

                    st.success("Summary Generated Successfully!")
                    st.write(output_summary)

        except Exception as e:
            st.error(f"An error occurred: {e}")
