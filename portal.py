import streamlit as st
import os

# --- Load Gemini API Key from Streamlit Secrets ---
GEMINI_KEY = st.secrets.get("api_keys", {}).get("google_gemini_key")

if GEMINI_KEY:
    st.write("✅ Gemini API key loaded successfully.")
else:
    st.warning("⚠️ Gemini API key not found. Check .streamlit/secrets.toml file.")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI/ML/DL Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Progress Tracking ---
if 'progress' not in st.session_state:
    st.session_state.progress = {}

# --- 24 Weeks Curriculum Data ---
weeks = [
    {"id": 1, "title": "Python Basics", "youtube": "https://www.youtube.com/watch?v=_uQrJ0TkZlc", "project": "Basic Calculator in Python", "dataset": "https://www.kaggle.com/datasets/uciml/iris"},
    {"id": 2, "title": "Data Structures in Python", "youtube": "https://www.youtube.com/watch?v=R-HLU9Fl5ug", "project": "Student Record System", "dataset": "https://www.kaggle.com/datasets/uciml/student-performance"},
    {"id": 3, "title": "Numpy & Pandas", "youtube": "https://www.youtube.com/watch?v=vmEHCJofslg", "project": "Data Cleaning Project", "dataset": "https://www.kaggle.com/datasets/c/house-prices-advanced-regression-techniques"},
    {"id": 4, "title": "Data Visualization", "youtube": "https://www.youtube.com/watch?v=DAQNHzOcO5A", "project": "COVID Data Visualization Dashboard", "dataset": "https://www.kaggle.com/datasets/imdevskp/corona-virus-report"},
    {"id": 5, "title": "Statistics & Probability", "youtube": "https://www.youtube.com/watch?v=xxpc-HPKN28", "project": "Predict Dice Roll Probabilities", "dataset": "https://www.kaggle.com/datasets/abcsds/pokemon"},
    {"id": 6, "title": "Machine Learning Overview", "youtube": "https://www.youtube.com/watch?v=GwIo3gDZCVQ", "project": "ML Pipeline Overview", "dataset": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"},
    {"id": 7, "title": "Linear Regression", "youtube": "https://www.youtube.com/watch?v=E5RjzSK0fvY", "project": "Predict House Prices", "dataset": "https://www.kaggle.com/datasets/shree1992/housedata"},
    {"id": 8, "title": "Logistic Regression", "youtube": "https://www.youtube.com/watch?v=yIYKR4sgzI8", "project": "Predict Heart Disease", "dataset": "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"},
    {"id": 9, "title": "Decision Trees", "youtube": "https://www.youtube.com/watch?v=ZVR2Way4nwQ", "project": "Predict Loan Approval", "dataset": "https://www.kaggle.com/datasets/ninzaami/loan-predication"},
    {"id": 10, "title": "Random Forest & Ensemble Learning", "youtube": "https://www.youtube.com/watch?v=J4Wdy0Wc_xQ", "project": "Customer Churn Prediction", "dataset": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"},
    {"id": 11, "title": "KNN & SVM", "youtube": "https://www.youtube.com/watch?v=4HKqjENq9OU", "project": "Classify Handwritten Digits", "dataset": "https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"},
    {"id": 12, "title": "Clustering (Unsupervised Learning)", "youtube": "https://www.youtube.com/watch?v=4b5d3muPQmA", "project": "Customer Segmentation", "dataset": "https://www.kaggle.com/datasets/shwetabh123/mall-customers"},
    {"id": 13, "title": "Neural Networks Basics", "youtube": "https://www.youtube.com/watch?v=aircAruvnKk", "project": "Digit Recognition with NN", "dataset": "https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"},
    {"id": 14, "title": "Deep Learning Frameworks", "youtube": "https://www.youtube.com/watch?v=V_xro1bcAuA", "project": "Build First Model with TensorFlow", "dataset": "https://www.kaggle.com/datasets/zalando-research/fashionmnist"},
    {"id": 15, "title": "CNNs", "youtube": "https://www.youtube.com/watch?v=YRhxdVk_sIs", "project": "Face Mask Detection", "dataset": "https://www.kaggle.com/datasets/omkargurav/face-mask-dataset"},
    {"id": 16, "title": "Transfer Learning", "youtube": "https://www.youtube.com/watch?v=G6r1AirexYk", "project": "Image Classification using VGG16", "dataset": "https://www.kaggle.com/datasets/puneet6060/intel-image-classification"},
    {"id": 17, "title": "RNNs", "youtube": "https://www.youtube.com/watch?v=WCUNPb-5EYI", "project": "Text Generation", "dataset": "https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification"},
    {"id": 18, "title": "LSTMs", "youtube": "https://www.youtube.com/watch?v=8HyCNIVRbSU", "project": "Stock Price Prediction", "dataset": "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs"},
    {"id": 19, "title": "Transformers & BERT", "youtube": "https://www.youtube.com/watch?v=U0s0f995w14", "project": "Text Sentiment Classification", "dataset": "https://www.kaggle.com/datasets/kazanova/sentiment140"},
    {"id": 20, "title": "Computer Vision with OpenCV", "youtube": "https://www.youtube.com/watch?v=oXlwWbU8l2o", "project": "Women Safety CCTV Alert System", "dataset": "https://www.kaggle.com/datasets/rajat95sawhney/women-safety"},
    {"id": 21, "title": "NLP with Hugging Face", "youtube": "https://www.youtube.com/watch?v=kCc8FmEb1nY", "project": "Chatbot with Transformers", "dataset": "https://www.kaggle.com/datasets/gauravduttakiit/chatbot-intents-dataset"},
    {"id": 22, "title": "GANs", "youtube": "https://www.youtube.com/watch?v=8L11aMN5KY8", "project": "Generate Anime Faces", "dataset": "https://www.kaggle.com/datasets/splcher/animefacedataset"},
    {"id": 23, "title": "Reinforcement Learning", "youtube": "https://www.youtube.com/watch?v=Mut_u40Sqz4", "project": "CartPole AI Agent", "dataset": "https://gym.openai.com/envs/CartPole-v1/"},
    {"id": 24, "title": "Capstone Project & Deployment", "youtube": "https://www.youtube.com/watch?v=tPYj3fFJGjk", "project": "Deploy ML Model with Streamlit", "dataset": "https://www.kaggle.com/datasets/"}
]

# --- Sidebar Navigation ---
st.sidebar.title("🧠 AI / ML / DL Roadmap")
page = st.sidebar.radio("Navigation", ["🏡 Home", "📚 Lessons", "🏗️ Projects", "🤖 Chatbot"])

# --- Home Page ---
if page == "🏡 Home":
    st.title("✨ AI / ML / DL 24-Week Portal")
    completed = len(st.session_state.progress)
    total = len(weeks)
    percent = int((completed / total) * 100) if total > 0 else 0
    st.metric("Progress", f"{completed}/{total}", f"{percent}%")
    st.progress(percent)

# --- Lessons Page ---
elif page == "📚 Lessons":
    for w in weeks:
        sid = str(w["id"])
        complete = sid in st.session_state.progress
        status = "✅" if complete else "⏳"
        with st.expander(f"{status} Week {w['id']}: {w['title']}", expanded=not complete):
            st.video(w["youtube"])
            st.write(f"**Project:** {w['project']}")
            st.link_button("Dataset", w["dataset"])
            if complete:
                if st.button("Mark Incomplete", key=f"unmark{sid}"):
                    st.session_state.progress.pop(sid)
                    st.rerun()
            else:
                if st.button("Mark Complete", key=f"mark{sid}"):
                    st.session_state.progress[sid] = True
                    st.rerun()

# --- Projects Page ---
elif page == "🏗️ Projects":
    st.title("🧩 Project Catalogue")
    for w in weeks:
        st.markdown(f"**Week {w['id']} - {w['title']}**")
        st.write(f"Project: {w['project']}")
        st.link_button("Dataset", w["dataset"])

# --- Chatbot Page ---
elif page == "🤖 Chatbot":
    st.title("🤖 Gemini AI Chatbot")
    st.caption("Powered by Google Gemini API")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I’m your AI tutor."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask something about AI, ML, or DL")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if GEMINI_KEY:
            response = f"(Gemini API simulated response to: '{prompt}')"
        else:
            response = f"(Gemini key missing. Placeholder response for: '{prompt}')"

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
