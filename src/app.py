import streamlit as st

try:
    from .inference import SentimentPredictor
except ImportError:
    from inference import SentimentPredictor


@st.cache_resource
def load_predictor():
    return SentimentPredictor()


def main():
    st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon=":movie_camera:")
    st.title("Movie Review Sentiment Analyzer")
    st.write("Enter a movie review to predict whether the sentiment is positive or negative.")

    predictor = load_predictor()
    review = st.text_area("Review", height=200, placeholder="Type or paste a movie review here...")

    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter some text before running a prediction.")
            return

        result = predictor.predict(review)
        st.subheader(f"Prediction: {result.label}")
        st.progress(int(result.confidence * 100))
        st.write(f"Model confidence: {result.confidence * 100:.2f}%")
        st.write(f"Positive sentiment probability: {result.positive_probability * 100:.2f}%")


if __name__ == "__main__":
    main()
