import os
import certifi
import whisper
import streamlit as st
import tempfile
import time
from openai import OpenAI
import re

# Set SSL certs for OpenAI on some systems
os.environ['SSL_CERT_FILE'] = certifi.where()

# Initialize OpenAI client for NVIDIA
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Load Whisper once
@st.cache_resource
def load_transcription_model():
    return whisper.load_model("tiny")

# Save transcript
def save_transcript(text):
    os.makedirs("results", exist_ok=True)
    with open("results/saved_transcript.txt", "a") as f:
        f.write(f"\n\n---\nTranscript:\n{text}")

# Extract a simple summary
def summarize_transcript(text):
    sentences = re.split(r'[.?!]', text)
    summary = ' '.join(sentences[:3]).strip()
    return summary + "..." if len(sentences) > 3 else summary

# Extract topics via keyword frequency or pattern
def extract_topics(text):
    keywords = ['client', 'feedback', 'AI', 'team', 'timeline', 'planning', 'problem', 'issue', 'solution', 'feature']
    found = [kw for kw in keywords if kw.lower() in text.lower()]
    return list(set(found))[:5]

# Generate prompt suggestions based on topics
def generate_prompt_suggestions_ai(summary: str, n: int = 5) -> list[str]:
    prompt = f"""
You are an AI assistant helping a user explore a meeting transcript. Given the following summary:
\"\"\"{summary}\"\"\"
Suggest {n} thoughtful, specific follow-up questions the user could ask to better understand or analyze this conversation. Return only the questions in a numbered list.
"""
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": "You generate useful follow-up questions from summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=512
    )
    raw = response.choices[0].message.content
    # Extract the questions from numbered list
    questions = [line.strip()[3:] for line in raw.split("\n") if line.strip().startswith(tuple("1234567890"))]
    return questions

# Transcribe uploaded audio
def transcribe_audio(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        temp_path = tmp.name
    model = load_transcription_model()
    result = model.transcribe(temp_path)
    return result["text"]

# Generate LLM response
def get_llm_response(transcript, query):
    context = f"Transcript:\n{transcript}\n\nUser Query: {query}"
    completion = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers questions based on the provided transcript."},
            {"role": "user", "content": context}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=True
    )

    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    return full_response

# Initialize session state
def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

# Main app
def main():
    st.title("🎙️ AI Note-Taking Assistant")
    init_state()

    # Commenting out the recording feature for now
    # """
    # # Audio recording feature (commented out for now)
    # st.subheader("🎤 Audio Recording Feature (Commented Out)")

    # # Recording buttons
    # if 'is_recording' not in st.session_state:
    #     st.session_state.is_recording = False

    # if st.button("Start Recording"):
    #     st.session_state.is_recording = True
    #     # Start recording logic (you can add the code here when ready)
    #     st.write("Recording in progress...")

    # if st.button("Stop Recording"):
    #     st.session_state.is_recording = False
    #     # Stop recording logic (you can add the code here when ready)
    #     st.write("Recording stopped.")

    # if st.session_state.is_recording:
    #     # Display audio wave, timer, etc. while recording
    #     st.write("Recording...")
    # else:
    #     st.write("Click 'Start Recording' to begin.")
    # """

    uploaded_file = st.file_uploader("Upload Audio/Video File", type=["mp3", "mp4", "wav", "m4a"])

    if uploaded_file:
        with st.spinner("Transcribing... ⏳"):
            progress = st.progress(0)
            time.sleep(0.3)
            transcript = transcribe_audio(uploaded_file)
            st.session_state.transcript = transcript
            save_transcript(transcript)
            time.sleep(0.3)
            progress.progress(100)

        st.success("Transcription Complete ✅")
        st.subheader("📄 Transcript")
        st.text_area("Full Transcript", transcript, height=200)
        st.download_button("📥 Download Transcript", transcript, file_name="transcript.txt")

        # Display insights
        summary = summarize_transcript(transcript)
        topics = extract_topics(transcript)
        suggestions = generate_prompt_suggestions_ai(summary)

        st.markdown("---")
        st.markdown("### 🧠 Insights & Suggestions")
        st.markdown(f"**Topics Detected:** {', '.join(topics)}")
        st.markdown("**💡 Suggested Follow-Up Questions:**")

        cols = st.columns(2)
        for i, prompt in enumerate(suggestions):
            if cols[i % 2].button(prompt):
                st.session_state.user_query = prompt

        st.markdown("---")
        st.text_input("Ask a question based on the transcript:", key="user_query")

    # If user enters a query
    if st.session_state.user_query and st.session_state.transcript:
        with st.spinner("Generating response... 🧠"):
            response = get_llm_response(st.session_state.transcript, st.session_state.user_query)
            st.session_state.chat_history.append((st.session_state.user_query, response))

        st.subheader("📝 Response")
        st.markdown(f"**Answer:**\n\n{response}")

    # Chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for idx, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{idx}: {q}"):
                st.markdown(f"**Answer:**\n{a}")

        chat_log = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        st.download_button("📥 Download Chat History", chat_log, file_name="chat_history.txt")

if __name__ == "__main__":
    main()
