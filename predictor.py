def predict_rating(cost, price_range, votes):
    """
    Simple function to predict restaurant rating
    """
    model = joblib.load('best_random_forest.pkl')
    new_data = pd.DataFrame({
        'Average Cost for two': [cost],
        'Price range': [price_range],
        'Votes': [votes]
    })
    return model.predict(new_data)[0]
