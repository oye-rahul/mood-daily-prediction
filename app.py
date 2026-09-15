import streamlit as st
import pickle
import string

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Mood & Emotion Classifier",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for Premium Design & Aesthetics
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }

    /* Gradient Header Banner */
    .header-container {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.4);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }

    /* Result Card Styling */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
    }

    .emoji-display {
        font-size: 5rem;
        margin-bottom: 0.5rem;
        animation: pulse 2s infinite alternate;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.08); }
    }

    .emotion-title {
        font-size: 2rem;
        font-weight: 800;
        text-transform: capitalize;
        margin-bottom: 0.25rem;
    }

    .confidence-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(129, 140, 248, 0.3);
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Section Subheaders */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Quick Try Buttons Styling */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.2s ease;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #818CF8;
        color: #818CF8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved ML Resources
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    with open('logistic_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open('emotion_mapping.pkl', 'rb') as map_file:
        mapping = pickle.load(map_file)
    return model, vectorizer, mapping

try:
    model, vectorizer, mapping = load_resources()
except FileNotFoundError:
    st.error("⚠️ Saved model components (`logistic_model.pkl`, `tfidf_vectorizer.pkl`, `emotion_mapping.pkl`) not found. Please run the model training notebook first.")
    st.stop()

# Emotion Visual & Styling Dictionary
EMOTION_META = {
    'sadness': {'emoji': '😢', 'color': '#60A5FA', 'desc': 'Feeling down, melancholy, or disappointed.'},
    'joy': {'emoji': '😊', 'color': '#F59E0B', 'desc': 'Feeling happy, cheerful, or fulfilled.'},
    'love': {'emoji': '❤️', 'color': '#EC4899', 'desc': 'Feeling affectionate, warm, or compassionate.'},
    'anger': {'emoji': '😡', 'color': '#EF4444', 'desc': 'Feeling frustrated, annoyed, or enraged.'},
    'fear': {'emoji': '📁', 'color': '#A855F7', 'desc': 'Feeling anxious, nervous, or threatened.'},
    'surprise': {'emoji': '😮', 'color': '#10B981', 'desc': 'Feeling astonished, amazed, or startled.'}
}

# Preprocessing Function (matches exact notebook pipeline)
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Model Info")
    st.markdown("""
    **Model**: Logistic Regression  
    **Vectorization**: TF-IDF  
    **Accuracy**: ~89%  
    **Classes**: Sadness, Joy, Love, Anger, Fear, Surprise  
    """)
    st.divider()
    st.markdown("### 💡 How it works")
    st.write("""
    1. Text is normalized & cleaned (lowercase, punctuation, numbers & non-ASCII removed).
    2. Transformed into numerical vectors via TF-IDF.
    3. Evaluated across 6 emotion classes with probability distribution.
    """)

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">🎭 Mood & Emotion AI</div>
    <div class="header-subtitle">Analyze the subtle emotional tone of any message instantly</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sample Quick Inputs
# ---------------------------------------------------------
st.markdown("<div class=\"section-header\">💡 Quick Try Examples</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
preset_text = ""

with col1:
    if st.button("😊 Sunny Day", use_container_width=True):
        preset_text = "I had such a wonderful morning, everything went smoothly and I feel so glad!"

with col2:
    if st.button("😡 Frustrated", use_container_width=True):
        preset_text = "I cannot believe they canceled my flight at the last minute, this is completely unacceptable!"

with col3:
    if st.button("📁 Nervous", use_container_width=True):
        preset_text = "I have a big presentation in front of 500 people tomorrow and my hands are trembling."

# ---------------------------------------------------------
# Main Input Form
# ---------------------------------------------------------
input_val = preset_text if preset_text else ""

user_input = st.text_area(
    "Enter your sentence or paragraph:",
    value=input_val,
    height=130,
    placeholder="Type or paste your text here (e.g., 'I felt so relieved when I finally finished my project...')"
)

classify_clicked = st.button("✨ Predict Emotion", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Inference & Results Section
# ---------------------------------------------------------
if classify_clicked or preset_text:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to classify!")
    else:
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])

        # Predict Class & Probabilities
        prediction_id = model.predict(vec_input)[0]
        emotion_name = mapping.get(prediction_id, "unknown").lower()

        # Class probabilities
        probabilities = model.predict_proba(vec_input)[0]

        meta = EMOTION_META.get(emotion_name, {'emoji': '✨', 'color': '#818CF8', 'desc': 'Detected emotion.'})
        confidence = probabilities[prediction_id] * 100

        # Primary Result Display Card
        st.markdown(f"""
        <div class="result-card">
            <div class="emoji-display">{meta['emoji']}</div>
            <div class="emotion-title" style="color: {meta['color']};">{emotion_name}</div>
            <div class="confidence-badge">Confidence: {confidence:.1f}%</div>
            <p style="margin-top: 1rem; color: rgba(255,255,255,0.7);">{meta['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability Breakdown Section
        st.markdown("<div class=\"section-header\">📊 Probability Breakdown</div>", unsafe_allow_html=True)

        # Map all class probabilities
        prob_pairs = []
        for idx, prob in enumerate(probabilities):
            name = mapping.get(idx, f"Class {idx}").lower()
            prob_pairs.append((name, prob))

        # Sort by highest probability
        prob_pairs.sort(key=lambda x: x[1], reverse=True)

        for name, prob in prob_pairs:
            e_meta = EMOTION_META.get(name, {'emoji': '✨', 'color': '#818CF8'})
            p_percentage = int(prob * 100)

            p_col1, p_col2, p_col3 = st.columns([2, 6, 2])
            with p_col1:
                st.write(f"{e_meta['emoji']} **{name.capitalize()}**")
            with p_col2:
                st.progress(prob)
            with p_col3:
                st.write(f"**{p_percentage}%**")
