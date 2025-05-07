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

# def get_df(df_reviews, df_restaurants, nb_reviews_per_restaurant):
#     #drop reviews with 0 stars
#     df_restaurants = df_restaurants[df_restaurants['stars'] > 0]
#     df_resto = df_restaurants[df_restaurants['categories'].str.contains('Restaurants', na=False)]
#     df_reviews_restaurants = df_reviews[df_reviews['business_id'].isin(df_resto['business_id'])]
#     df_reviews_interesting = df_reviews_restaurants[df_reviews_restaurants['useful'] > 2]
#     # df_reviews_use_1 = df_reviews_interesting[df_reviews_interesting['useful'] == 1]
#     df_restaurents_stat = df_reviews_restaurants.groupby('business_id').agg(
#         rating_count=('stars', 'count'),
#     )
# 
#     for i in range(1, 6):
#         df_ratio = df_reviews_restaurants[df_reviews_restaurants['stars'] == i].groupby('business_id').size() / df_restaurents_stat['rating_count']
#         df_restaurents_stat[f'ratio_{i}_stars'] = df_ratio
#     
#     # Fill NaNs with 0 (for businesses that never got some specific ratings)
#     df_restaurents_stat = df_restaurents_stat.fillna(0)
#     df_restaurents_stat = df_restaurents_stat.reset_index()
#     print(df_restaurents_stat.head())
# 
#     dfs = []
#     for company_id in df_restaurents_stat['business_id']:
#         print("company_id:", company_id)
#         df_review_filtered = df_reviews_interesting[df_reviews_interesting['business_id'] == company_id]
#         for star in range(1, 6):
#             df_reviews_interesting_star = df_review_filtered[df_review_filtered['stars'] == star]
#             df_reviews_interesting_star = df_reviews_interesting_star.sort_values(by='date', ascending=False)
#             ratio = df_restaurents_stat.loc[df_restaurents_stat['business_id'] == company_id, f'ratio_{star}_stars'].iloc[0]
#             df_reviews_interesting_star = df_reviews_interesting_star.head(int(nb_reviews_per_restaurant * ratio))
#             dfs.append(df_reviews_interesting_star)
#     df = pd.concat(dfs)
#     df = df.drop_duplicates(subset=['review_id'])  
#     return df

# def get_df(df_reviews, df_restaurants, nb_reviews_per_restaurant):
#     # Keep only restaurants with >0 stars and relevant category
#     df_restaurants = df_restaurants[df_restaurants['stars'] > 0]
#     df_resto = df_restaurants[df_restaurants['categories'].str.contains('Restaurants', na=False)]
# 
#     # Filter reviews for restaurants and those marked useful
#     df_reviews_restaurants = df_reviews[df_reviews['business_id'].isin(df_resto['business_id'])]
#     df_reviews_interesting = df_reviews_restaurants[df_reviews_restaurants['useful'] > 2]
# 
#     # Compute the number of total reviews per business
#     total_counts = df_reviews_restaurants.groupby('business_id')['stars'].count().rename('rating_count')
# 
#     # Compute per-star count and convert to ratio
#     star_counts = df_reviews_restaurants.groupby(['business_id', 'stars']).size().unstack(fill_value=0)
#     ratios = star_counts.div(total_counts, axis=0).fillna(0)
#     ratios.columns = [f'ratio_{col}_stars' for col in ratios.columns]
# 
#     # Merge ratio info
#     df_restaurents_stat = ratios.reset_index()
# 
#     # Prepare a list to collect selected reviews
#     dfs = []
# 
#     # Iterate per restaurant
#     for _, row in df_restaurents_stat.iterrows():
#         business_id = row['business_id']
#         print("business_id:", business_id)
#         df_reviews_biz = df_reviews_interesting[df_reviews_interesting['business_id'] == business_id]
# 
#         for star in range(1, 6):
#             star_ratio_col = f'ratio_{star}_stars'
#             if star_ratio_col not in row:
#                 continue
#             ratio = row[star_ratio_col]
#             if ratio == 0:
#                 continue
# 
#             n_to_keep = int(nb_reviews_per_restaurant * ratio)
#             df_reviews_star = df_reviews_biz[df_reviews_biz['stars'] == star]
#             df_reviews_star = df_reviews_star.sort_values(by='date', ascending=False).head(n_to_keep)
#             dfs.append(df_reviews_star)
# 
#     # Concatenate and remove duplicates
#     df_result = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='review_id')
#     return df_result


import pandas as pd
import numpy as np

def get_df(df_reviews, df_restaurants, nb_reviews_per_restaurant):
    # Filter to only restaurants with >0 stars and category "Restaurants"
    df_restaurants = df_restaurants[df_restaurants['stars'] > 0]
    df_resto = df_restaurants[df_restaurants['categories'].str.contains('Restaurants', na=False)]

    # Filter reviews for restaurants and "useful" > 2
    df_reviews_restaurants = df_reviews[df_reviews['business_id'].isin(df_resto['business_id'])]
    df_reviews_interesting = df_reviews_restaurants[df_reviews_restaurants['useful'] > 2]

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

    return df_final




df = get_df(df_review, df_business, 10)
df.to_pickle('../data_set/reviews2.pkl')
