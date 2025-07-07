from utils import load_data, preprocess_data, test_model, load_model, train_model, save_model

df = load_data('data/cancer patient data sets.csv')
print(df.shape)

x_train, x_test, y_train, y_test = preprocess_data(df)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# train model
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()

# lr = train_model(lr, x_train, y_train)
# save_model(lr, 'models\logistic_regression_fitted.joblib')


# test model
model = load_model('models\logistic_regression_fitted.joblib')
print(test_model(model, x_test, y_test))


