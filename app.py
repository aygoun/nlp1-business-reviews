import streamlit as st
import json
import pandas as pd
import random
import time
from PIL import Image
import io
import base64
import numpy as np
from datetime import datetime
import os

# from text_gen.transformer.Qwen_prompting import QwenReviewGenerator
# from text_gen.transformer.Custom_prompting import CustomReviewGenerator
# from text_gen.transformer.Pegasus_prompting import PegasusReviewGenerator
from classification.NaiveBayes.StarsAnalyzerNB import SentimentAnalyzerNB
from classification.Transformer.Transformer import ClassificationTransformer

from text_gen.transformer.Gpt2_inferring import GPT2ReviewGenerator

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

# Function to load and cache the model
@st.cache_resource
def load_generative_models():
    """
    Load the Qwen review generator model
    """
    gpt2 = GPT2ReviewGenerator("text_gen/transformer/gpt2_review_finetuned_200")

    return gpt2

@st.cache_resource
def load_classification_models():
    """
    Load the Qwen review generator model
    """
    bayes = SentimentAnalyzerNB("")
    bayes.init()
    transformer = ClassificationTransformer("models/Classification_Transformer")


    return bayes, transformer

# Function to load business data
@st.cache_data
def load_business_data(file_path="data_set/yelp_academic_dataset_business.json"):
    """
    Load business data from the Yelp academic dataset JSON file
    """
    # For demo purposes, return mock data
    # In production, replace with actual file loading:
    ids = ['ytynqOUb3hjKeJfRj5Tshw','xlMQBBt9wrtahdqiRDcVSg',
 'GBTPC53ZrG1ZBY3DT8Mbcw', '6a4gLLFSgr-Q6CZXDLzBGQ',
 'j-qtdD55OLfSqfsWuQTDJg', 'PP3BBaVxZLcJU54uP_wL6Q',
 'IkY2ticzHEn4QFn8hQLSWg', 'iSRTaT9WngzB8JJ2YKJUig',
 '_C7QiQQc47AOEv4PE3Kong', 'ac1AeYqs8Z4_e2X5M3if2A',
 'sTPueJEwcRDj7ZJmG7okYA', 'RQAF6a0akMiot5lZZnMNNw',
 '1b5mnK8bMnnju_cvU65GqQ', '8pqdJjefYq-a9IBSJJmKwA',
 'mhrW9O0O5hXGXGnEYBVoag', 'skY6r8WAkYqpV7_TxNm23w',
 '9PZxjhTIU7OgPIzuGi89Ew', '0RuvlgTnKFbX3IK0ZOOocA',
 'VAeEXLbEcI9Emt9KGYq9aA', 'oBNrLz4EDhiscSlbOl8uAw',
 'SZU9c8V2GuREDN5KgyHFJw', '6ajnOk0GcY9xbb5Ocaw8Gw',
 'U3grYFIeu6RgAAQgdriHww', 'GXFMD0Z4jEVZBCsbPf4CTQ',
 'OWOOc0YjU_kioLeEgo5VCA', 'VQcCL9PiNL_wkGf-uF3fjg',
 'DcBLYSvOuWcNReolRVr12A', 'XnQ84ylyAZwh-XfHGGNBbQ',
 'J0joPXxmN-_9Lzafspqdbw', 'gTC8IQ_i8zXytWSly3Ttvg',
 '3YqUe2FTCQr0pPVK8oCv6Q', 'AGlh4ZDv6jnoiYfz7At9mw',
 'ww3YJXu5c18aGZXWmm00qg', 'Y2Pfil51rNvTd_lFHwzb_g',
 'dsfRniRgfbDjC8os848B6A', 'C9K3579SJgLPp0oAOM29wg',
 '_ab50qdWOk0DdB6XOrBitw', 'yPSejq3_erxo9zdVYTBnZA',
 'L5LLN0RafiV1Z9cddzvuCw', 'EagkHaaC-kUozD3MPzbRIw',
 'RLlOK2fL6xU1sfIPiP2QBw', 'VaO-VW3e1kARkU9bP1E7Fw',
 'ChlcxTEoWBQJXJ2Xb2vm5g', 'VVH6k9-ycttH3TV_lk5WfQ']
    with open(file_path, 'r') as f:
        businesses = []
        for line in f:
            businesses.append(json.loads(line))
        
        # Return only lines with business ids in the list ids
        res = []
        for business in businesses:
            if business["business_id"] in ids:
                res.append(business)
        return res
    
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
def generate_review(business, model_type, nb_stars):
    """
    Generate a review based on the business and model type
    This is a placeholder - replace with actual model integration
    """
    # Simulate processing time
    with st.spinner("Generating review..."):
        time.sleep(1)  # Simulate processing time
    
    business_name = business["name"]
    # business_type = business["categories"].split(",")[0].strip()
    
    review = model_type.gen_review(business_name, nb_stars)
    # review = f"This is a generated review for {business_name}. The service was excellent and the food was delicious!"  # Simulate generated review
    # sentiment_score = random.uniform(1.0, 5.0)  # Simulate sentiment score
    sentiment_score = st.session_state.class_model.predict_stars(review)
    
    return review, float(sentiment_score)

# Function to analyze sentiment
def analyze_sentiment(review, model_type):
    """
    Analyze sentiment of a review
    This is a placeholder - replace with actual sentiment analysis model
    """
    # Simulate processing time
    with st.spinner("Analyzing sentiment..."):
        time.sleep(0.5)  # Simulate processing time
    
    # For demo purposes, return slightly different results based on model
    return st.session_state.class_model.predict_stars(st.session_state.generated_review)

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

# Function to generate dataset for classification models
def generate_classification_dataset(min_rating, reviews_per_restaurant):
    """
    Generate a dataset for classification models
    [...]
    """
    chunksize = 1000
    dfs = []
    for df in pd.read_json('../data_set/yelp_academic_dataset_review.json', lines=True, chunksize=chunksize):
        dfs.append(df)
    df_reviews = pd.concat(dfs, ignore_index=True)

    dfs = []
    for df in pd.read_json('../data_set/yelp_academic_dataset_business.json', lines=True, chunksize=chunksize):
        dfs.append(df)
    df_restaurants = pd.concat(dfs, ignore_index=True)

    df_resto = df_restaurants[df_restaurants['categories'].str.contains('Restaurants', na=False)]
    df_reviews_restaurants = df_reviews[df_reviews['business_id'].isin(df_resto['business_id'])]
    df_review_filtered = df_reviews_restaurants.groupby('business_id').agg(
        average_rating=('stars', 'mean'),
        rating_count=('stars', 'count'),
        median_rating=('stars', 'median')
    ).reset_index()
    df_sup_t = df_review_filtered[df_review_filtered['rating_count'] > min_rating]
    dfs = []
    nb_reviews = 0
    for i in range(1,6):
        temp = df_sup_t[(df_sup_t['median_rating'] == i) & (df_sup_t['average_rating'] >= i - 0.5) & (df_sup_t['average_rating'] <= i + 0.5)]
        temp = temp.head(reviews_per_restaurant // 5)
        nb_reviews += temp['rating_count'].sum()
        dfs.append(temp)
    df_sup_t = pd.concat(dfs)

    df_low_t = df_review_filtered[df_review_filtered['rating_count'] <= min_rating]
    dfs = []
    for i in range(1,6):
        offset = 10
        target = nb_reviews // (5 - i + 1)
        temp = df_low_t[(df_low_t['median_rating'] == i) & (df_low_t['average_rating'] >= i - 0.5) & (df_low_t['average_rating'] <= i + 0.5)]
        res = temp.head(reviews_per_restaurant // 5 + offset)
        while target > res['rating_count'].sum() and res.shape[0] < reviews_per_restaurant // 5 + offset:
            print("in iteration:", i)
            print("actual count:", res['rating_count'].sum())
            print("actual target:", target)
            offset += 10
            res = temp.head(reviews_per_restaurant // 5 + offset)
        nb_reviews -= res['rating_count'].sum()
        dfs.append(temp)
    df_low_t = pd.concat(dfs)

    df_res = pd.concat([df_sup_t, df_low_t])
    df = df_reviews_restaurants[df_reviews_restaurants['business_id'].isin(df_res['business_id'])]
   
    # Creates and saves a CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_set/classification_dataset_{timestamp}.pkl"
    df.to_pickle(filename, index=False)
    
    return df, filename

# Function to generate dataset for generative models
def generate_generative_dataset(reviews_per_restaurant, length_max = 100, nb_restaurants = 52000):
    """
    Generate a dataset for generative models
    [...]
    """

    chunksize = 1000
    dfs = []
    for df in pd.read_json('../data_set/yelp_academic_dataset_review.json', lines=True, chunksize=chunksize):
        dfs.append(df)
    df_reviews = pd.concat(dfs, ignore_index=True)

    dfs = []
    for df in pd.read_json('../data_set/yelp_academic_dataset_business.json', lines=True, chunksize=chunksize):
        dfs.append(df)
    df_restaurants = pd.concat(dfs, ignore_index=True)

    df_restaurants = df_restaurants[df_restaurants['stars'] > 0]
    df_resto = df_restaurants[df_restaurants['categories'].str.contains('Restaurants', na=False)]
    df_resto = df_resto.sort_values(by='review_count', ascending=False)
    df_resto = df_resto.head(nb_restaurants)

    # Filter reviews for restaurants and "useful" > 2
    df_reviews_restaurants = df_reviews[df_reviews['business_id'].isin(df_resto['business_id'])]
    df_reviews_interesting = df_reviews_restaurants[df_reviews_restaurants['useful'] > 2]
    df_reviews_interesting = df_reviews_interesting[df_reviews_interesting['text'].str.len() < length_max]

    # Compute total review counts and star-specific ratios
    total_counts = df_reviews_restaurants.groupby('business_id')['stars'].count().rename('rating_count')
    star_counts = df_reviews_restaurants.groupby(['business_id', 'stars']).size().unstack(fill_value=0)

    # Calculate ratios
    ratios = star_counts.div(total_counts, axis=0).fillna(0)
    ratios.columns = [f'ratio_{col}_stars' for col in ratios.columns]
    df_restaurents_stat = ratios.reset_index()
    
    # Merge ratios into interesting reviews
    df_reviews_interesting = df_reviews_interesting.merge(df_restaurents_stat, on='business_id', how='inner')

    # Compute for each row how many reviews to keep
    def calculate_keep_count(row):
        ratio = row[f'ratio_{int(row.stars)}_stars']
        return int(reviews_per_restaurant * ratio)

    df_reviews_interesting['keep_count'] = df_reviews_interesting.apply(calculate_keep_count, axis=1)

    # Sort reviews by date within business_id and stars
    df_reviews_interesting.sort_values(['business_id', 'stars','useful', 'date'], ascending=[True, True, False, False], inplace=True)

    # Assign rank within each (business_id, stars) group
    df_reviews_interesting['rank'] = df_reviews_interesting.groupby(['business_id', 'stars']).cumcount()

    # Keep only top N reviews per group
    df_final = df_reviews_interesting[df_reviews_interesting['rank'] < df_reviews_interesting['keep_count']]
    df_final = df_final.drop_duplicates(subset='review_id').reset_index(drop=True)
    df_final = df_final.drop(columns=['keep_count', 'rank', 'ratio_1_stars', 'ratio_2_stars', 'ratio_3_stars', 'ratio_4_stars', 'ratio_5_stars', 'cool', 'funny', 'user_id'])

    df_final = df_final.merge(df_resto[['business_id', 'name']], on='business_id', how='inner')
    df_final = df_final.rename(columns={'name': 'restaurant_name'})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_set/generative_dataset_{timestamp}.pkl"
    df_final.to_pickle(filename, index=False)
    
    return df_final, filename

# Main application
def main():
    # Initialize session state if not present
    if 'businesses' not in st.session_state:
        st.session_state.businesses = load_business_data()
    if 'generated_review' not in st.session_state:
        st.session_state.generated_review = ""
    if 'sentiment_score' not in st.session_state:
        st.session_state.sentiment_score = 0
    if 'classidication_models' not in st.session_state:
        bayes, transformer_model = load_classification_models()
        st.session_state.classification_models = {"transformer": transformer_model, "bayes": bayes}
    if 'class_model' not in st.session_state:
        st.session_state.class_model = st.session_state.classification_models["transformer"]
    if 'generative_models' not in st.session_state:
        gpt2_model = load_generative_models()
        st.session_state.generative_models = {"gpt2": gpt2_model}
    if 'gen_model' not in st.session_state:
        st.session_state.gen_model = st.session_state.generative_models["gpt2"]
    # New session state variables
    if 'classification_filename' not in st.session_state:
        st.session_state.classification_filename = "data_set/reviews.pkl"
    if 'generative_filename' not in st.session_state:
        st.session_state.generative_filename = "data_set/reviews2.pkl"
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
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
            # "qwen": "Qwen Transformer Generator",
            # "pegasus": "Pegasus Transformer Generator",
            # "custom": "Custom Transformer Generator"
            "gpt2": "GPT-2 Transformer Generator"
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
            # "basic": "Basic Sentiment Analysis",
            # "advanced": "Advanced Sentiment Analysis"
            "bayes": "Naive Bayes Sentiment Analysis",
            "transformer": "Transformer Sentiment Analysis",
        }
        selected_sentiment_model = st.selectbox(
            "Select Sentiment Model",
            options=list(sentiment_model_options.keys()),
            format_func=lambda x: sentiment_model_options[x],
            index=0
        )
        
        analyze_button = st.button("Re-analyze Sentiment", use_container_width=True)

        st.divider()

        # Dataset Generation section
        st.subheader("Dataset Generation")

        # Classification dataset parameters
        st.markdown("### Classification Dataset")
        min_rating = st.slider("Minimum Rating", min_value=20, max_value=2000, value=500, step=1)
        class_reviews_per_restaurant = st.number_input("Reviews per Restaurant (Classification)", 
                                                   min_value=1, max_value=2000, value=10)

        # Button to generate classification dataset
        if st.button("Generate Classification Dataset", use_container_width=True):
            dataset, filename = generate_classification_dataset(min_rating, class_reviews_per_restaurant)
            st.session_state.classification_filename = filename
            st.success(f"Classification dataset generated and saved as {filename}")

        # Generative dataset parameters
        st.markdown("### Generative Dataset")
        gen_reviews_per_restaurant = st.number_input("Reviews per Restaurant (Generative)", 
                                                min_value=1, max_value=500, value=50)
        gen_reviews_max_len = st.number_input("Maximum Length of Reviews",
                                             min_value=1, max_value=1000, value=100)
        nb_restaurant = st.number_input("Number of Restaurant",
                                             min_value=1, max_value=52000, value=20000)

        # Button to generate generative dataset
        if st.button("Generate Generative Dataset", use_container_width=True):
            dataset, filename = generate_generative_dataset(gen_reviews_per_restaurant, gen_reviews_max_len, nb_restaurant)
            st.session_state.generative_filename = filename
            st.success(f"Generative dataset generated and saved as {filename}")
    
    tab1, tab2 = st.tabs(["Review Generation & Analysis", "Dataset Generation"])

    with tab1:
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
                st.session_state.class_model = st.session_state.classification_models[selected_sentiment_model]
                st.session_state.gen_model = st.session_state.generative_models[selected_model]
                review, score = generate_review(selected_business, st.session_state.generative_models[selected_model], selected_business['stars'])
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
                    st.session_state.class_model = st.session_state.classification_models[selected_sentiment_model]
                    print(st.session_state.class_model)
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
    
    with tab2:
        st.markdown("## Dataset Generation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Classification Dataset")
            # Download button
            st.info("Generate a classification dataset using the controls in the sidebar.")
            st.markdown("""
            This dataset is optimized for classification models with parameters:
            - Minimum rating threshold
            - Number of reviews per restaurant
            """)
            st.download_button(
                label="Download Classification Dataset",
                file_name=st.session_state.classification_filename,
                data=pd.read_pickle(st.session_state.classification_filename).to_csv(index=False),  # Convert DataFrame to CSV string
                mime='text/csv',
            )
        
        with col2:
            st.markdown("### Generative Dataset")
            st.info("Generate a dataset for generative models using the controls in the sidebar.")
            st.markdown("""
            This dataset is optimized for generative models with parameters:
            - Number of reviews per restaurant
            - Maximum length of reviews
            - Number of restaurants
            """)
            st.download_button(
                label="Download Generative Dataset",
                file_name=st.session_state.generative_filename,
                data=pd.read_pickle(st.session_state.generative_filename).to_csv(index=False),  # Convert DataFrame to CSV string
                mime='text/csv',
            )
        
        # Information about integrating with your models
        st.markdown("## How to Integrate with Your Models")
        st.markdown("""
        To integrate these datasets with your existing models:
        
        1. Download the generated datasets using the buttons above
        2. Feed these datasets into your classification and generative models
        3. Update the `generate_review()` and `analyze_sentiment()` functions in this app to call your actual models
        
        """)
    
    # Footer with note about mock data
    st.markdown("---")
    st.caption("Note: This is a demo with mock data. In a real implementation, you would load data from yelp_academic_business.json.")

if __name__ == "__main__":
    main()
