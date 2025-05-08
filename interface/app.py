import streamlit as st
import json
import pandas as pd
import random
import time
from PIL import Image
import io
import base64
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Yelp Reviews Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add some custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .business-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .review-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-style: italic;
    }
    .sentiment-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    .comparison-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
    }
    .star-rating {
        color: gold;
        font-size: 1.5rem;
    }
    .empty-star {
        color: #d3d3d3;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("Yelp Reviews Analysis Interface")

# Function to load business data
@st.cache_data
def load_business_data(file_path="../data_set/yelp_academic_dataset_business.json"):
    """
    Load business data from the Yelp academic dataset JSON file
    """
    # For demo purposes, return mock data
    # In production, replace with actual file loading:
    with open(file_path, 'r') as f:
        businesses = []
        for line in f:
            businesses.append(json.loads(line))
        return businesses
    
    # Mock data for demonstration
    # return [
    #     {"business_id": "Pns2l4eNsfO8kk83dixA6A", "name": "Abby Rappoport, LAC, CMQ", "address": "1616 Chapala St, Ste 2", "city": "Santa Barbara", "state": "CA", "postal_code": "93101", "latitude": 34.4266787, "longitude": -119.7111968, "stars": 5.0, "review_count": 7, "is_open": 0, "attributes": {"ByAppointmentOnly": "True"}, "categories": "Doctors, Traditional Chinese Medicine, Naturopathic/Holistic, Acupuncture, Health & Medical, Nutritionists", "hours": None},
    #     {"business_id": "mpf3x-BjTdTEA3yCZrAYPw", "name": "The UPS Store", "address": "87 Grasso Plaza Shopping Center", "city": "Affton", "state": "MO", "postal_code": "63123", "latitude": 38.551126, "longitude": -90.335695, "stars": 3.0, "review_count": 15, "is_open": 1, "attributes": {"BusinessAcceptsCreditCards": "True"}, "categories": "Shipping Centers, Local Services, Notaries, Printing Services", "hours": {"Monday": "8:0-18:30", "Tuesday": "8:0-18:30", "Wednesday": "8:0-18:30", "Thursday": "8:0-18:30", "Friday": "8:0-18:30", "Saturday": "8:0-16:0"}},
    #     {"business_id": "UFrWirKiKi_TAnsVWINQQ", "name": "Target", "address": "5255 E Broadway Blvd", "city": "Tucson", "state": "AZ", "postal_code": "85711", "latitude": 32.223236, "longitude": -110.880452, "stars": 3.5, "review_count": 22, "is_open": 1, "attributes": {"BusinessParking": "{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}", "RestaurantsPriceRange2": "2", "BusinessAcceptsCreditCards": "True", "WheelchairAccessible": "True", "RestaurantsTakeOut": "True"}, "categories": "Department Stores, Shopping, Fashion, Home & Garden, Electronics, Furniture Stores", "hours": {"Monday": "8:0-22:0", "Tuesday": "8:0-22:0", "Wednesday": "8:0-22:0", "Thursday": "8:0-22:0", "Friday": "8:0-22:0", "Saturday": "8:0-22:0", "Sunday": "8:0-22:0"}},
    #     {"business_id": "jxTXpA6tYjFRUfQ9Q3qSHQ", "name": "Starbucks", "address": "7845 Highland Ave", "city": "San Diego", "state": "CA", "postal_code": "92115", "latitude": 32.7612157, "longitude": -117.0526974, "stars": 4.0, "review_count": 134, "is_open": 1, "attributes": {"WiFi": "free", "BusinessAcceptsCreditCards": "True"}, "categories": "Coffee & Tea, Food, Cafes", "hours": {"Monday": "5:0-22:0", "Tuesday": "5:0-22:0", "Wednesday": "5:0-22:0", "Thursday": "5:0-22:0", "Friday": "5:0-22:0", "Saturday": "5:30-22:0", "Sunday": "5:30-22:0"}},
    #     {"business_id": "cGDaQK_7cSnlL4XAQWcJTQ", "name": "Subway", "address": "6702 Melrose Ave", "city": "Los Angeles", "state": "CA", "postal_code": "90038", "latitude": 34.0837813, "longitude": -118.3375041, "stars": 2.5, "review_count": 86, "is_open": 1, "attributes": {"RestaurantsTakeOut": "True", "RestaurantsPriceRange2": "1", "BusinessAcceptsCreditCards": "True"}, "categories": "Fast Food, Restaurants, Sandwiches", "hours": {"Monday": "7:0-22:0", "Tuesday": "7:0-22:0", "Wednesday": "7:0-22:0", "Thursday": "7:0-22:0", "Friday": "7:0-22:0", "Saturday": "8:0-22:0", "Sunday": "9:0-21:0"}}
    # ]

# Function to render star rating
def render_star_rating(score):
    """
    Render a star rating based on the score
    """
    full_stars = int(score)
    half_star = score - full_stars >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    stars_html = ""
    for _ in range(full_stars):
        stars_html += "★"
    if half_star:
        stars_html += "½"
    for _ in range(empty_stars):
        stars_html += "☆"
    
    return f'<span class="star-rating">{stars_html}</span> <span>{score:.1f}</span>'

# Function to generate a review
def generate_review(business, model_type):
    """
    Generate a review based on the business and model type
    This is a placeholder - replace with actual model integration
    """
    # Simulate processing time
    with st.spinner("Generating review..."):
        time.sleep(1)  # Simulate processing time
    
    business_name = business["name"]
    business_type = business["categories"].split(",")[0].strip()
    
    if model_type == "positive":
        review = f"I absolutely loved my experience at {business_name}! As a {business_type}, they really exceed expectations. The staff was incredibly helpful and the service was top-notch. I would definitely recommend this place to anyone looking for quality service in this area."
        sentiment_score = 4.8
    elif model_type == "critical":
        review = f"My visit to {business_name} was somewhat disappointing. While they offer standard {business_type} services, I found the overall experience to be mediocre. The waiting time was longer than expected and the value for money wasn't great. There's definitely room for improvement."
        sentiment_score = 2.3
    else:  # default
        review = f"{business_name} provides {business_type} services that meet the standard expectations. My experience was generally positive with a few minor issues. The staff was professional and the facilities were clean. It's a solid option if you're in the area and need these services."
        sentiment_score = 3.5
    
    return review, sentiment_score

# Function to analyze sentiment
def analyze_sentiment(review, model_type):
    """
    Analyze sentiment of a review
    This is a placeholder - replace with actual sentiment analysis model
    """
    # Simulate processing time
    with st.spinner("Analyzing sentiment..."):
        time.sleep(0.8)  # Simulate processing time
    
    # For demo purposes, return slightly different results based on model
    if model_type == "advanced":
        # More nuanced scores
        if "loved" in review.lower() or "excellent" in review.lower():
            return 4.9
        elif "disappointing" in review.lower() or "mediocre" in review.lower():
            return 2.1
        else:
            return 3.7
    else:  # basic
        # More rounded scores
        if "loved" in review.lower() or "excellent" in review.lower():
            return 5.0
        elif "disappointing" in review.lower() or "mediocre" in review.lower():
            return 2.0
        else:
            return 3.5

# Function to get sentiment description
def get_sentiment_description(score):
    if score >= 4.5:
        return "Very Positive"
    elif score >= 3.5:
        return "Positive"
    elif score >= 2.5:
        return "Neutral"
    elif score >= 1.5:
        return "Negative"
    else:
        return "Very Negative"

# Main application
def main():
    # Initialize session state if not present
    if 'businesses' not in st.session_state:
        st.session_state.businesses = load_business_data()
    if 'generated_review' not in st.session_state:
        st.session_state.generated_review = ""
    if 'sentiment_score' not in st.session_state:
        st.session_state.sentiment_score = 0
    
    # Sidebar for controls
    with st.sidebar:
        # st.header("Controls")
        
        # File uploader for custom data (not functional in this demo)
        # uploaded_file = st.file_uploader("Upload Yelp business JSON (optional)", type=["json"])
        # if uploaded_file:
        #     st.warning("File upload functionality is simulated in this demo.")
        
        st.divider()
        
        # Business selection
        business_options = {b["name"] + " - " + b["city"] + ", " + b["state"]: b["business_id"] 
                         for b in st.session_state.businesses}
        selected_business_id = st.selectbox(
            "Select Business", 
            options=list(business_options.keys()),
            format_func=lambda x: x,
            index=0
        )
        
        selected_business = next(
            (b for b in st.session_state.businesses 
             if b["business_id"] == business_options[selected_business_id]), 
            None
        )
        
        st.divider()
        
        # Model selection
        st.subheader("Review Generation")
        model_options = {
            "default": "Default Review Generator",
            "positive": "Positive Biased Generator",
            "critical": "Critical Review Generator"
        }
        selected_model = st.selectbox(
            "Select Review Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        generate_button = st.button("Generate Review", use_container_width=True)
        
        st.divider()
        
        # Sentiment analysis model selection
        st.subheader("Sentiment Analysis")
        sentiment_model_options = {
            "basic": "Basic Sentiment Analysis",
            "advanced": "Advanced Sentiment Analysis"
        }
        selected_sentiment_model = st.selectbox(
            "Select Sentiment Model",
            options=list(sentiment_model_options.keys()),
            format_func=lambda x: sentiment_model_options[x],
            index=0
        )
        
        analyze_button = st.button("Re-analyze Sentiment", use_container_width=True)
    
    # Main content area
    if selected_business:
        # Business information card
        st.markdown("<h2>Business Information</h2>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="business-card">
            <h3>{selected_business["name"]}</h3>
            <p>{selected_business["address"]}, {selected_business["city"]}, {selected_business["state"]} {selected_business["postal_code"]}</p>
            <p><strong>Categories:</strong> {selected_business["categories"]}</p>
            <p><strong>Actual Yelp Rating:</strong> {render_star_rating(selected_business["stars"])} ({selected_business["review_count"]} reviews)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate review if button was clicked
        if generate_button or (st.session_state.generated_review == "" and selected_business):
            review, score = generate_review(selected_business, selected_model)
            st.session_state.generated_review = review
            st.session_state.sentiment_score = score
        
        # Display generated review
        if st.session_state.generated_review:
            st.markdown("<h2>Generated Review</h2>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="review-card">
                {st.session_state.generated_review}
            </div>
            """, unsafe_allow_html=True)
            
            # Re-analyze sentiment if button was clicked
            if analyze_button:
                st.session_state.sentiment_score = analyze_sentiment(
                    st.session_state.generated_review, 
                    selected_sentiment_model
                )
            
            # Display sentiment analysis
            st.markdown("<h2>Sentiment Analysis</h2>", unsafe_allow_html=True)
            sentiment_description = get_sentiment_description(st.session_state.sentiment_score)
            st.markdown(f"""
            <div class="sentiment-card">
                <p><strong>Sentiment Score:</strong> {render_star_rating(st.session_state.sentiment_score)}</p>
                <p><strong>Sentiment:</strong> {sentiment_description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Rating comparison
            st.markdown("<h2>Rating Comparison</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="comparison-card">
                    <div>
                        <p><strong>Actual Yelp Rating:</strong></p>
                        <p>{render_star_rating(selected_business["stars"])}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="comparison-card">
                    <div>
                        <p><strong>Generated Review Rating:</strong></p>
                        <p>{render_star_rating(st.session_state.sentiment_score)}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Please select a business from the sidebar to begin.")
    
    # Footer with note about mock data
    st.markdown("---")
    st.caption("Note: This is a demo with mock data. In a real implementation, you would load data from yelp_academic_business.json.")

if __name__ == "__main__":
    main()