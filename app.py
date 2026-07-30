"""
app.py
------
Streamlit web application for the Auto Email / Ticket Categorizer.

Loads the trained Naive Bayes model + TF-IDF vectorizer and provides:
    - A text area to paste in a new support ticket
    - Predicted department category
    - Confidence score
    - "Needs Human Review" flag for low-confidence predictions
    - Priority detection (High / Normal) based on urgency keywords
    - Ready-to-use sample tickets for quick testing

Run:
    streamlit run app.py
"""

import os
import re
import string

import joblib
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
CONFIDENCE_THRESHOLD = 0.60  # Below this -> "Needs Human Review"

# Keywords that indicate an urgent / high priority ticket
URGENT_KEYWORDS = [
    "urgent",
    "critical",
    "server down",
    "payment failed",
    "not working",
    "asap",
    "immediately",
]

# Keywords most associated with each category, used to build a
# human-readable explanation for the prediction.
CATEGORY_KEYWORDS = {
    "Technical": [
        "login", "error", "crash", "password", "bug", "server",
        "install", "update", "app", "software", "device", "network",
        "connect", "loading", "sync", "freeze",
    ],
    "Billing": [
        "invoice", "payment", "refund", "charge", "subscription",
        "billing", "plan", "gst", "amount", "gateway", "deducted",
        "cancel", "price", "receipt",
    ],
    "HR": [
        "leave", "salary", "payroll", "onboarding", "resignation",
        "policy", "attendance", "hr", "employee", "offer letter",
        "benefits", "holiday", "appraisal",
    ],
    "General": [
        "information", "question", "query", "help", "feedback",
        "suggestion", "general", "contact", "hours", "location",
    ],
}

# A handful of ready-made tickets so users/reviewers can test instantly
SAMPLE_TICKETS = [
    "I was charged twice for my subscription this month",
    "The app keeps crashing every time I try to log in",
    "I want to apply for maternity leave, please share the policy",
    "What are your customer support working hours",
    "Server is down and payment failed for multiple customers, please help immediately",
    "Need urgent help, my account got locked after password reset",
    "Can you update my billing address on file",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean raw ticket text (lowercase, remove punctuation, extra spaces)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load the trained model and TF-IDF vectorizer from disk.

    Returns:
        (model, vectorizer) tuple, or (None, None) if files are missing.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def detect_priority(raw_text: str) -> str:
    """
    Detect ticket priority based on the presence of urgency keywords.

    Args:
        raw_text: The original (uncleaned) ticket text, so multi-word
            phrases like "server down" can be matched reliably.

    Returns:
        "High" if any urgent keyword is found, else "Normal".
    """
    text_lower = raw_text.lower()
    for keyword in URGENT_KEYWORDS:
        if keyword in text_lower:
            return "High"
    return "Normal"


def build_explanation(cleaned_text: str, predicted_category: str) -> str:
    """
    Build a human-readable explanation of why a ticket was classified
    into a given category, based on matched keywords.

    Args:
        cleaned_text: The cleaned ticket text.
        predicted_category: The category predicted by the model.

    Returns:
        A short explanation string.
    """
    keywords = CATEGORY_KEYWORDS.get(predicted_category, [])
    matched = [kw for kw in keywords if kw in cleaned_text]

    if matched:
        matched_str = ", ".join(matched[:5])
        return (
            f"The ticket contains words such as {matched_str}, which are "
            f"commonly associated with {predicted_category} requests."
        )
    return (
        f"The model identified overall language patterns in the ticket "
        f"that most closely match the {predicted_category} category."
    )


def predict_ticket(raw_text: str, model, vectorizer) -> dict:
    """
    Predict the category of a new support ticket.

    Args:
        raw_text: The raw ticket text entered by the user.
        model: Trained MultinomialNB model.
        vectorizer: Fitted TfidfVectorizer.

    Returns:
        A dictionary containing:
            - category: predicted category (or "Needs Human Review")
            - confidence: confidence percentage (float)
            - explanation: human-readable explanation string
            - priority: "High" or "Normal"
            - needs_review: bool
    """
    cleaned = clean_text(raw_text)

    if not cleaned:
        return {
            "category": "Unknown",
            "confidence": 0.0,
            "explanation": "The ticket text is empty or invalid after cleaning.",
            "priority": "Normal",
            "needs_review": True,
        }

    vector = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(vector)[0]
    classes = model.classes_

    best_idx = int(np.argmax(probabilities))
    predicted_category = classes[best_idx]
    confidence = float(probabilities[best_idx])

    priority = detect_priority(raw_text)
    needs_review = confidence < CONFIDENCE_THRESHOLD

    explanation = build_explanation(cleaned, predicted_category)

    return {
        "category": predicted_category,
        "confidence": confidence,
        "explanation": explanation,
        "priority": priority,
        "needs_review": needs_review,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Auto Ticket Categorizer",
        page_icon="🎫",
        layout="centered",
    )

    # --- Custom styling ---
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            text-align: center;
            color: #6b7280;
            margin-bottom: 1.8rem;
        }
        .result-card {
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-title">🎫 Auto Email / Ticket Categorizer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">NLP-powered support ticket routing: '
        "Billing · Technical · HR · General</div>",
        unsafe_allow_html=True,
    )

    model, vectorizer = load_artifacts()

    if model is None or vectorizer is None:
        st.error(
            "Model files not found. Please run `python train.py` first to "
            "generate `model.pkl` and `vectorizer.pkl`."
        )
        st.stop()

    # --- Sidebar: sample tickets ---
    with st.sidebar:
        st.header("📋 Sample Tickets")
        st.caption("Click any sample to load it into the input box.")
        for i, sample in enumerate(SAMPLE_TICKETS):
            if st.button(sample, key=f"sample_{i}", use_container_width=True):
                st.session_state["ticket_input"] = sample

        st.markdown("---")
        st.caption(
            "Human review threshold: "
            f"**{int(CONFIDENCE_THRESHOLD * 100)}%** confidence"
        )
        st.caption(
            "Priority keywords: urgent, critical, server down, "
            "payment failed, not working, asap, immediately"
        )

    # --- Main input ---
    ticket_text = st.text_area(
        "Enter a customer support ticket",
        key="ticket_input",
        height=140,
        placeholder="e.g. My payment failed but the amount was deducted from my account...",
    )

    predict_clicked = st.button("🔍 Predict Category", type="primary", use_container_width=True)

    if predict_clicked:
        if not ticket_text or not ticket_text.strip():
            st.warning("Please enter a ticket description before predicting.")
        else:
            with st.spinner("Analyzing ticket..."):
                result = predict_ticket(ticket_text, model, vectorizer)

            st.markdown("### Result")

            col1, col2, col3 = st.columns(3)
            with col1:
                if result["needs_review"]:
                    st.metric("Category", "Needs Review")
                else:
                    st.metric("Category", result["category"])
            with col2:
                st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            with col3:
                priority_display = (
                    f"🔴 {result['priority']}"
                    if result["priority"] == "High"
                    else f"🟢 {result['priority']}"
                )
                st.metric("Priority", priority_display)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            if result["needs_review"]:
                st.warning(
                    "⚠️ **Needs Human Review** — confidence is below "
                    f"{int(CONFIDENCE_THRESHOLD * 100)}%. This ticket was not "
                    "auto-assigned to a department."
                )
                st.write(
                    f"Model's best guess was **{result['category']}**, "
                    f"but confidence was too low to auto-route it."
                )
            else:
                st.success(f"✅ **Predicted Category:** {result['category']}")

            st.write("**Explanation:**")
            st.info(result["explanation"])

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Built for the AI/ML Internship Assessment — "
        "TF-IDF + Multinomial Naive Bayes text classifier."
    )


if __name__ == "__main__":
    main()
