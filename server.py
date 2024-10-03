import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.valid_locations = {
            "Albuquerque, New Mexico",
            "Carlsbad, California",
            "Chula Vista, California",
            "Colorado Springs, Colorado",
            "Denver, Colorado",
            "El Cajon, California",
            "El Paso, Texas",
            "Escondido, California",
            "Fresno, California",
            "La Mesa, California",
            "Las Vegas, Nevada",
            "Los Angeles, California",
            "Oceanside, California",
            "Phoenix, Arizona",
            "Sacramento, California",
            "Salt Lake City, Utah",
            "Salt Lake City, Utah",
            "San Diego, California",
            "Tucson, Arizona"
        }

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores
    
    def filter_reviews(self, location, start_date, end_date):
        filtered_reviews = []
        for review in reviews:
            review_date = datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S")
            if (not location or review['Location'] == location) and \
               (not start_date or review_date >= start_date) and \
               (not end_date or review_date <= end_date):
                filtered_reviews.append(review)
        filtered_reviews.sort(key=lambda x: self.analyze_sentiment(x['ReviewBody'])['compound'], reverse=True)
        return filtered_reviews
    
    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            parsed = parse_qs(environ['QUERY_STRING'])
            location = parsed.get('location', [None])[0]
            start_date_str = parsed.get('start_date', [None])[0]
            end_date_str = parsed.get('end_date', [None])[0]

            start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None

            if location and location not in self.valid_locations:
                response_body = json.dumps({"error": "Invalid location"}, indent=2).encode("utf-8")
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [response_body]

            filtered_reviews = self.filter_reviews(location, start_date, end_date)
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except (ValueError):
                request_body_size = 0

            request_body = environ['wsgi.input'].read(request_body_size)
            review_data = parse_qs(request_body.decode('utf-8'))

            review_body = review_data.get('ReviewBody', [None])[0]
            location = review_data.get('Location', [None])[0]

            if not review_body or not location:
                response_body = json.dumps({"error": "Missing ReviewBody or Location"}, indent=2).encode("utf-8")
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [response_body]

            if location not in self.valid_locations:
                response_body = json.dumps({"error": "Invalid location"}, indent=2).encode("utf-8")
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [response_body]

            sentiment_scores = self.analyze_sentiment(review_body)
            review_id = str(uuid.uuid4())
            review_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            new_review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": review_date,
                "sentiment": sentiment_scores
            }

            reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()