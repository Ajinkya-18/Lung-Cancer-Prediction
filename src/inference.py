from utils import get_test_data, load_model

# load the fitted (trained) column transformer and ordinal encoder
col_transformer = load_model('models/col_transformer_fitted.joblib')
ord_enc = load_model('models/ordinal_encoder_fitted.joblib')

# load the fitted model to make inference (predictions)
model = load_model('models/random_forest_classifier_fitted.joblib')

# get test data 
test_data = get_test_data()
# print(test_data.shape)

# transform the test data
test_data_scaled = col_transformer.transform(test_data)

# pass the test data to model for predictions
predictions = model.predict(test_data_scaled)
y_hat = ord_enc.inverse_transform(predictions.reshape(-1, 1))[0][0]

# show the test data and its predictions
print(test_data)
print('\n')
print(f'Lung Disease susceptibility: {y_hat}')

