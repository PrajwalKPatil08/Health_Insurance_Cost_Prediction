from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)

    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    return regressor, r2_train, r2_test

def predict_cost(model, input_data):
    input_data_reshaped = [input_data]
    return model.predict(input_data_reshaped)[0]
