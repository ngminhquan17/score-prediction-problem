import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = pd.read_csv("StudentScore.xls")
target = "math score"
#print(data.corr(numeric_only=True))

#Phân chia dữ liệu theo chiều dọc và ngang
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
#print(x_train, x_test, y_train, y_test)
#print(data["parental level of education"].unique())

#Xử lý numerical
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()) 
])

#xử lý ordinal + boolen
education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",  "master's degree"]
gender_values = ["female", "male"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()

ordinal_transfomer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('scaler', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values])) 
])

#Xử lý nordinal
nominal_transfomer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('scaler', OneHotEncoder(sparse_output=False)) 
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", numerical_transformer, ["reading score", "writing score"]),
    ("ord_feature", ordinal_transfomer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nominal_transfomer, ["race/ethnicity"]),
])

'''result = nordinal_transfomer.fit_transform(x_train[["race/ethnicity"]])
for i, j in zip(x_train[["race/ethnicity"]].values, result):
    print("Before {}, After {}".format(i,j))'''


#Chạy mô hình
reg = Pipeline(steps=[
    ('preprocessor', preprocessor), #Tiền xử lý
    ("model", RandomForestRegressor())
])

# Nếu sử dụng lazy thì dùng code này
'''reg = Pipeline(steps=[
    ('preprocessor', preprocessor), #Tiền xử lý
    #("model", LinearRegression())
])

x_train = reg.fit_transform(x_train)
x_test = reg.transform(x_test)'''

params = {
    "model__n_estimators": [50, 100, 200],
    "model__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "model__max_features": ["sqrt", "log2", None],
    "preprocessor__num_features__imputer__strategy" : ["median", "mean"],
}

#grid_reg = GridSearchCV(reg, param_grid=params, cv=6, verbose=1, scoring="r2", n_jobs=-1)
grid_reg = RandomizedSearchCV(reg, param_distributions=params, cv=6, verbose=1, scoring="r2", n_jobs=-1, n_iter=20)
grid_reg.fit(x_train, y_train)

#Kiểm tra
#grid_reg.fit(x_train, y_train)
y_predict = grid_reg.predict(x_test)

'''
for i, j in zip(y_predict, y_test):
    print("Predict: {}. Actual: {}".format(i, j))'''

print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))

# Random Forest 
# MSE: 35.84889327777778
# MAE: 4.809816666666666
# R2: 0.8270074622995386

# Linear Regression cho kết quả tốt hơn
# MSE: 28.857953703152642
# MAE: 4.291302546899144
# R2: 0.8607429633805347

'''reg = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = reg.fit(x_train, x_test, y_train, y_test)
print(models)'''

# grean search



