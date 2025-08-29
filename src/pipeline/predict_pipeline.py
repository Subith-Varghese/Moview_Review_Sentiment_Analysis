import sys
from src.components.predictor import Predictor
from src.utils.logger import logger

def run_prediction():
    try:
        # Initialize predictor
        predictor = Predictor()

        # Take user input
        review = input("\n📝 Enter a review to analyze sentiment: ").strip()
        if not review:
            logger.error("❌ No input provided. Please enter a valid review.")
            sys.exit(1)

        # Get prediction result
        result = predictor.predict(review)
        print(result)
        
        logger.info("✅ Prediction pipeline executed successfully.")

    except Exception as e:
        logger.exception(f"❌ Unexpected error in prediction pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_prediction()
