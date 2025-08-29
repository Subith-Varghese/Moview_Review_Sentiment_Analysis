from flask import Flask, render_template, request
from src.components.predictor import Predictor
from src.utils.logger import logger

# Initialize Flask app
app = Flask(__name__)

# Initialize the Predictor class
predictor = Predictor()

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_result = None
    pred_prob  = None
    review_text = None
    rating = None
    stars = None

    if request.method == "POST":
        try:
            # Get the review entered by the user
            review_text = request.form.get("review")

            if not review_text or review_text.strip() == "":
                sentiment_result = "⚠️ Please enter a valid review!"
            else:
                # Predict using Predictor
                stars, pred_prob, rating = predictor.predict(review_text)
                
                # Sentiment classification based on rating
                if rating >= 3.5:
                    sentiment_result = "Positive"
                elif rating <= 1.5:
                    sentiment_result = "Negative"
                else:
                    sentiment_result = "Neutral"


                logger.info(f"🎯 Review: {review_text} | Rating: {rating} | Stars: {stars} | Prob: {pred_prob}")

        except Exception as e:
            sentiment_result = "❌ Error occurred during prediction."
            logger.exception(f"Prediction error: {e}")

    return render_template(
        "home.html",
        sentiment=sentiment_result,
        probability=pred_prob,
        rating=rating,
        stars=stars,
        review=review_text
    )

if __name__ == "__main__":
    logger.info("🚀 Starting Flask App...")
    app.run(host="0.0.0.0", port=5000, debug=True)
