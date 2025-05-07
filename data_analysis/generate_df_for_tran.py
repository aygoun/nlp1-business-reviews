import pandas as pd

#Load Dataset of review
chunksize = 1000
dfs = []
for df in pd.read_json('../data_set/yelp_academic_dataset_review.json', lines=True, chunksize=chunksize):
    dfs.append(df)
df_review = pd.concat(dfs, ignore_index=True)

#Load Dataset of business
dfs = []
for df in pd.read_json('../data_set/yelp_academic_dataset_business.json', lines=True, chunksize=chunksize):
    dfs.append(df)
df_business = pd.concat(dfs, ignore_index=True)

def get_df(df_reviews, df_restaurants, nb_reviews_per_restaurant, length_max = 10000):
    # Filter to only restaurants with >0 stars and category "Restaurants"
    df_restaurants = df_restaurants[df_restaurants['stars'] > 0]
    df_resto = df_restaurants[df_restaurants['categories'].str.contains('Restaurants', na=False)]

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
        return int(nb_reviews_per_restaurant * ratio)

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

    return df_final

df = get_df(df_review, df_business, 20, 500)
df.to_pickle('../data_set/reviews2.pkl')