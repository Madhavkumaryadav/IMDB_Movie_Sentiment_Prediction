"""
MovieMind - IMDB Sentiment Analyzer
Run:  streamlit run main.py
"""

## Step 1 : Import libraries
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

MODEL_PATH = "simple_rnn_imdb_model.h5"
MODEL2_PATH = "simple_rnn_model.h5"


## Step 2 : Load the word index and the pre-trained Simple RNN model (once)
@st.cache_resource(show_spinner="Loading the IMDB word index...")
def load_word_index():
    return imdb.get_word_index()


@st.cache_resource(show_spinner="Loading the sentiment model...")
def load_sentiment_model():
    return load_model(MODEL2_PATH)


## Step 3 : Helper functions (logic unchanged from the original code)
def preprocess_text(text):
    word_index = load_word_index()
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    model = load_sentiment_model()
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input, verbose=0)

    score = float(prediction[0][0])

    sentiment = "Positive" if score >= 0.5 else "Negative"

    return sentiment, score


## Step 4 : Streamlit UI
# ---------------------------------------------------------------- Page config
st.set_page_config(
    page_title="MovieMind - IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ Constants
MIN_WORDS = 3
BORDERLINE_RANGE = (0.40, 0.60)
PLACEHOLDER = (
    "Example: The movie was fantastic, with outstanding acting "
    "and a brilliant storyline..."
)
EXAMPLES = {
    "Choose an example...": "",
    "Positive example": (
        "This movie was fantastic. The acting was outstanding, the story was "
        "brilliant and I was gripped from the first scene to the last. "
        "A wonderful film that I would happily watch again."
    ),
    "Negative example": (
        "A complete waste of time. The plot made no sense, the acting was "
        "terrible and the dialogue was painfully boring. I wanted to walk out "
        "halfway through."
    ),
    "Mixed example": (
        "The cinematography was beautiful and the soundtrack was great, but "
        "the story dragged and the ending was disappointing."
    ),
}

# ------------------------------------------------------------------------ CSS
CUSTOM_CSS = """
<style>
:root {
    --mm-navy: #0b1026;
    --mm-purple: #7c3aed;
    --mm-purple-dark: #6d28d9;
    --mm-text: #e8ecff;
    --mm-muted: #aab2d5;
    --mm-border: rgba(148, 163, 255, 0.22);
    --mm-green: #34d399;
    --mm-red: #f87171;
}
.stApp {
    background:
        radial-gradient(1100px 500px at 8% -10%, rgba(124, 58, 237, 0.28), transparent 60%),
        radial-gradient(900px 450px at 100% 0%, rgba(59, 130, 246, 0.20), transparent 60%),
        var(--mm-navy);
    color: var(--mm-text);
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }

/* Hero */
.mm-hero {
    padding: 2.2rem 2rem;
    border-radius: 20px;
    border: 1px solid var(--mm-border);
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.30), rgba(59, 130, 246, 0.14) 60%, rgba(20, 26, 58, 0.6));
    margin-bottom: 1.5rem;
}
.mm-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    border: 1px solid var(--mm-border);
    background: rgba(11, 16, 38, 0.55);
    color: var(--mm-muted);
    font-size: 0.85rem;
    margin-bottom: 0.9rem;
}
.mm-hero h1 {
    margin: 0; padding: 0;
    font-size: 2.6rem; line-height: 1.15; letter-spacing: -0.02em; color: #ffffff;
}
.mm-hero p { margin: 0.7rem 0 0 0; font-size: 1.15rem; color: var(--mm-muted); max-width: 46rem; }

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
    border-color: var(--mm-border);
    background: rgba(20, 26, 58, 0.55);
}

/* Text area */
.stTextArea textarea {
    font-size: 1.05rem; line-height: 1.55; border-radius: 12px;
    background: rgba(11, 16, 38, 0.85); color: var(--mm-text);
}
.stTextArea textarea::placeholder { color: #8d96bf; opacity: 1; }

/* Buttons */
.stButton > button {
    width: 100%; min-height: 3rem; border-radius: 12px;
    font-weight: 600; font-size: 1rem; border: 1px solid var(--mm-border);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--mm-purple-dark), var(--mm-purple));
    color: #ffffff; border: none;
}
.stButton > button:disabled { opacity: 0.5; cursor: not-allowed; }
.stButton > button:focus-visible,
.stTextArea textarea:focus-visible { outline: 3px solid #93c5fd; outline-offset: 2px; }

/* Result card */
.mm-result {
    border-radius: 16px; padding: 1.4rem 1.5rem;
    border: 1px solid var(--mm-border); border-left-width: 8px;
}
.mm-result.pos { border-left-color: var(--mm-green); background: rgba(52, 211, 153, 0.10); }
.mm-result.neg { border-left-color: var(--mm-red);   background: rgba(248, 113, 113, 0.10); }
.mm-result h2 { margin: 0 0 0.3rem 0; padding: 0; font-size: 1.8rem; }
.mm-result.pos h2 { color: var(--mm-green); }
.mm-result.neg h2 { color: var(--mm-red); }
.mm-result p { margin: 0.25rem 0; color: var(--mm-text); }
.mm-note { color: var(--mm-muted); font-size: 0.92rem; }

/* Score bar */
.mm-bar-title { margin: 1.1rem 0 0.4rem 0; font-weight: 600; }
.mm-bar {
    display: flex; height: 16px; border-radius: 999px; overflow: hidden;
    background: rgba(148, 163, 255, 0.15);
}
.mm-bar-neg { background: var(--mm-red); }
.mm-bar-pos { background: var(--mm-green); }
.mm-bar-legend {
    display: flex; justify-content: space-between;
    margin-top: 0.4rem; font-size: 0.92rem; color: var(--mm-muted);
}

/* Metrics */
div[data-testid="stMetric"] {
    padding: 0.7rem 0.9rem; border-radius: 12px;
    border: 1px solid var(--mm-border); background: rgba(11, 16, 38, 0.6);
}

@media (max-width: 640px) {
    .mm-hero { padding: 1.4rem 1.1rem; }
    .mm-hero h1 { font-size: 1.8rem; }
    .mm-hero p { font-size: 1rem; }
}
</style>
"""

HERO_HTML = """
<div class="mm-hero">
    <span class="mm-badge">🎬 Simple RNN &nbsp;|&nbsp; IMDB reviews</span>
    <h1>MovieMind — IMDB Sentiment Analyzer</h1>
    <p>Understand the emotion behind every movie review with AI.</p>
</div>
"""


# ------------------------------------------------------- UI helpers (no ML code)
def validate_review(text):
    """Basic input check before calling predict_sentiment()."""
    if not text.strip():
        return False, "Write or paste a movie review before analyzing."
    words = [w for w in text.split() if any(ch.isalnum() for ch in w)]
    if not words:
        return False, "No readable words found. Use letters or numbers, not only symbols."
    if len(words) < MIN_WORDS:
        return False, (
            f"This review is too short. Write at least {MIN_WORDS} words "
            "so the model has something to work with."
        )
    return True, ""


def init_state():
    st.session_state.setdefault("review_text", "")
    st.session_state.setdefault("example_choice", "Choose an example...")
    st.session_state.setdefault("result", None)


def on_example_change():
    st.session_state.review_text = EXAMPLES[st.session_state.example_choice]
    st.session_state.result = None


def on_clear():
    st.session_state.review_text = ""
    st.session_state.example_choice = "Choose an example..."
    st.session_state.result = None


def render_input_section():
    with st.container(border=True):
        st.subheader("Analyze a movie review")
        st.caption("Enter your review to discover its predicted sentiment.")

        st.selectbox(
            "Example review",
            options=list(EXAMPLES.keys()),
            key="example_choice",
            on_change=on_example_change,
        )
        st.text_area(
            "Movie review",
            key="review_text",
            height=200,
            placeholder=PLACEHOLDER,
            label_visibility="collapsed",
        )

        text = st.session_state.review_text
        st.caption(f"{len(text.split())} words, {len(text)} characters")

        analyze_col, clear_col = st.columns([2, 1])
        with analyze_col:
            analyze = st.button(
                "Analyze Sentiment",
                type="primary",
                disabled=not text.strip(),
                help="Enter a review to enable this button.",
            )
        with clear_col:
            st.button("Clear Review", on_click=on_clear)
    return text, analyze


def render_result(result, current_text):
    score = result["score"]
    positive = result["sentiment"] == "Positive"
    pos_pct = score * 100
    neg_pct = 100 - pos_pct

    message = (
        "The model predicts that this review expresses a positive opinion."
        if positive
        else "The model predicts that this review expresses a negative opinion."
    )
    borderline = ""
    if BORDERLINE_RANGE[0] <= score <= BORDERLINE_RANGE[1]:
        borderline = (
            "<p class='mm-note'>The score is close to the 50% decision threshold, "
            "so treat this prediction as uncertain.</p>"
        )

    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"""
            <div class="mm-result {'pos' if positive else 'neg'}" role="status">
                <h2>{result['sentiment']} Sentiment</h2>
                <p>{message}</p>
                {borderline}
                <p class="mm-bar-title">Model score by class</p>
                <div class="mm-bar" role="img"
                     aria-label="Negative {neg_pct:.1f} percent, positive {pos_pct:.1f} percent">
                    <div class="mm-bar-neg" style="width:{neg_pct:.2f}%"></div>
                    <div class="mm-bar-pos" style="width:{pos_pct:.2f}%"></div>
                </div>
                <div class="mm-bar-legend">
                    <span>Negative {neg_pct:.1f}%</span>
                    <span>Positive {pos_pct:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        m1, m2 = st.columns(2)
        m1.metric("Positive-class model score", f"{score:.4f}")
        m2.metric("As a percentage", f"{pos_pct:.1f}%")

    st.caption(
        "The score is the model's raw sigmoid output (estimated positive-class "
        "probability), not a calibrated confidence. Sarcasm, mixed opinions and "
        "unusual vocabulary can lead to wrong predictions."
    )
    if current_text != result["text"]:
        st.info("The review has changed since this result. Select Analyze Sentiment to update it.")


def render_how_it_works():
    with st.expander("How it works"):
        st.markdown(
            """
**Text preprocessing → sequence padding → Simple RNN → sentiment score**

1. **Preprocessing.** The review is lower-cased and split into words.
2. **Tokenization.** Each word is looked up in the IMDB word index and replaced by its
   integer id (shifted by 3 to leave room for padding, start and unknown tokens).
3. **Padding.** The sequence is padded with zeros to a fixed length of 500 tokens.
4. **Simple RNN prediction.** An embedding layer turns ids into vectors, the Simple RNN
   reads them in order, and a final sigmoid unit outputs a score between 0 and 1.
5. **Decision.** A score of 0.5 or higher is labelled *Positive*, anything lower *Negative*.
"""
        )


def render_model_info():
    with st.expander("Model information"):
        c1, c2 = st.columns(2)
        c1.metric("Architecture", "Simple RNN")
        c2.metric("Input length", "500 tokens")
        st.markdown(
            """
- **Model file:** `simple_rnn_imdb_model.h5`
- **Task:** binary classification (Positive / Negative)
- **Output:** one sigmoid score between 0 and 1
- **Training data:** IMDB movie reviews
"""
        )
        st.caption(
            "Not tuned for sarcasm, non-English text or other kinds of writing."
        )


# ------------------------------------------------------------------ Streamlit app
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)

init_state()
text, analyze = render_input_section()

if analyze:
    is_valid, message = validate_review(text)
    if not is_valid:
        st.error(message)
    else:
        cached = st.session_state.result
        # Reuse the stored result if this exact review was already analyzed.
        if cached is None or cached["text"] != text:
            try:
                with st.spinner("Analyzing review..."):
                    sentiment, score = predict_sentiment(text)  # your existing function
                st.session_state.result = {
                    "text": text,
                    "sentiment": sentiment,
                    "score": float(score),
                }
            except Exception as exc:
                st.session_state.result = None
                st.error(f"The review could not be analyzed: {exc}")

st.write("")
if st.session_state.result is not None:
    render_result(st.session_state.result, st.session_state.review_text)
else:
    st.info("Your result will appear here after you analyze a review.")

st.write("")
left_col, right_col = st.columns(2)
with left_col:
    render_how_it_works()
with right_col:
    render_model_info()