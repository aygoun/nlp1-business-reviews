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

def get_df(df_reviews, df_restaurants, review_threshold, restaurant_threshold=1000):
    df_resto = df_restaurants[df_restaurants['categories'].str.contains('Restaurants', na=False)]
    df_reviews_restaurants = df_reviews[df_reviews['business_id'].isin(df_resto['business_id'])]
    df_review_filtered = df_reviews_restaurants.groupby('business_id').agg(
        average_rating=('stars', 'mean'),
        rating_count=('stars', 'count'),
        median_rating=('stars', 'median')
    ).reset_index()
    df_sup_t = df_review_filtered[df_review_filtered['rating_count'] > review_threshold]
    dfs = []
    nb_reviews = 0
    for i in range(1,6):
        temp = df_sup_t[(df_sup_t['median_rating'] == i) & (df_sup_t['average_rating'] >= i - 0.5) & (df_sup_t['average_rating'] <= i + 0.5)]
        temp = temp.head(restaurant_threshold // 5)
        nb_reviews += temp['rating_count'].sum()
        dfs.append(temp)
    df_sup_t = pd.concat(dfs)

    df_low_t = df_review_filtered[df_review_filtered['rating_count'] <= review_threshold]
    dfs = []
    for i in range(1,6):
        offset = 10
        target = nb_reviews // (5 - i + 1)
        temp = df_low_t[(df_low_t['median_rating'] == i) & (df_low_t['average_rating'] >= i - 0.5) & (df_low_t['average_rating'] <= i + 0.5)]
        res = temp.head(restaurant_threshold // 5 + offset)
        while target > res['rating_count'].sum() and res.shape[0] < restaurant_threshold // 5 + offset:
            print("in iteration:", i)
            print("actual count:", res['rating_count'].sum())
            print("actual target:", target)
            offset += 10
            res = temp.head(restaurant_threshold // 5 + offset)
        nb_reviews -= res['rating_count'].sum()
        dfs.append(temp)
    df_low_t = pd.concat(dfs)

    df_res = pd.concat([df_sup_t, df_low_t])
    df = df_reviews_restaurants[df_reviews_restaurants['business_id'].isin(df_res['business_id'])]
    return df

df = get_df(df_review, df_business, 1000, 1000)
df.to_pickle('../data_set/reviews.pkl')
