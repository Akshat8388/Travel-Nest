from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)


first_recommendations_data = pd.read_csv("data/first_recommendations_systems.csv")
first_recommendations_data.rename(columns={
    'Google review rating': 'Google_review_rating',
    'Establishment Year': 'Establishment_Year',
    'Best Time to visit': 'Best_Time_to_visit',
    'Image URL': 'Image_URL'
}, inplace=True)

def first_recommendator():
    top_picks = first_recommendations_data.sort_values(by='recommendations', ascending=False).head(100).sample(5)
    return top_picks[["Name", "State", "City", "Google_review_rating", "Establishment_Year",
                      "Significance", "Best_Time_to_visit", "Image_URL"]].to_dict(orient='records')

second_recommendations_data = pd.read_csv("data/second_recommendations_systems.csv")
second_recommendations_data.rename(columns={
    'Google review rating': 'Google_review_rating',
    'Establishment Year': 'Establishment_Year',
    'Best Time to visit': 'Best_Time_to_visit',
    'Image URL': 'Image_URL'
}, inplace=True)

Places_name = sorted(second_recommendations_data["Name"].unique())
features_matrix_encoded = pd.read_csv("data/features_matrix_encoded_second_recommendations.csv")

from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(features_matrix_encoded)

def get_recommendations(place_name, second_recommendations_data, cos_sim):
    index = second_recommendations_data[second_recommendations_data["Name"] == place_name].index[0]
    similarity_scores = list(enumerate(cos_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    place_indices = [i[0] for i in similarity_scores]
    return second_recommendations_data.iloc[place_indices][["Name", "State", "City", "Google_review_rating",
                                                           "Establishment_Year", "Significance", "Best_Time_to_visit", "Image_URL"]].to_dict(orient='records')
    

third_recommendations_data = pd.read_csv("data/Third_recommendations_systems_data.csv")
third_recommendations_data.rename(columns={
    'Google review rating': 'Google_review_rating',
    'Establishment Year': 'Establishment_Year',
    'Best Time to visit': 'Best_Time_to_visit',
    'Image URL': 'Image_URL'
}, inplace=True)

def third_recommendations(region,third_recommendations_data):
    r = third_recommendations_data[third_recommendations_data['Zone']==region]
    cluster = r['Label'].unique()
    rem = pd.DataFrame()
    for c in cluster:
       l = r[r['Label']==c]
       top_destinations = l.nlargest(15, 'Google_review_rating')
       rem = pd.concat([rem, top_destinations], ignore_index=True)
    return rem[["Name", "State", "City", "Google_review_rating", "Establishment_Year",
                      "Significance", "Best_Time_to_visit", "Image_URL"]].sample(5).to_dict(orient='records')   
    
@app.route('/region', methods=['POST'])
def region():
    region = request.form.get('region')
    region_recommendation = third_recommendations(region,third_recommendations_data)
    return jsonify(region_recommendation)
    

    


@app.route('/')
def home():
    top_picks = first_recommendator()  
    return render_template('index.html', top_picks=top_picks, places_name=Places_name)

@app.route('/refresh-top-picks')
def refresh_top_picks():
    new_top_picks = first_recommendator()  
    return jsonify(new_top_picks) 

@app.route('/recommend', methods=['POST'])
def recommend():
    place_name = request.form.get('Places_name')  
    recommendations = get_recommendations(place_name, second_recommendations_data, cos_sim)
    return jsonify(recommendations) 
   

# if __name__ == '__main__':
#     app.run()
