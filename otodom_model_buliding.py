import pandas as pd
import numpy as np
import pandasql as ps
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# read the data into Pandas DataFrame
df = pd.read_csv('otodom_data2.csv', sep=';')
# print(df.head(5))

'''
################ DATA CLEANING ###################
'''

# clean price values -> remove ~, replace comas with dot
df.price = df.price.apply(lambda x: x.replace(',', '.').replace('~', ''))

# drop rows where price is not defined
df.drop(df[df.price == 'Zapytajocenę'].index, inplace=True)

# create a variable 'total_price' which includes price visible on a main site together with added rent
df.added_rent = df.added_rent.apply(lambda x: int(x.replace('[', '').replace(']', '')))
df.price = df.price.apply(lambda x: round(float(x)))
df['total_price'] = df.price + df.added_rent

# clean 'area' and 'year'
df.area = df.area.apply(lambda x: round(x))
df.year = df.year.apply(lambda x: (x.replace('[', '').replace(']', '')))
df.year = pd.to_numeric(df.year, errors='coerce')

# map features to new columns
df['zmywarka'] = df["features"].map(lambda x: 1 if 'zmywarka' in x else 0)
df['balkon'] = df["features"].map(lambda x: 1 if 'balkon' in x else 0)
df['piwnica'] = df["features"].map(lambda x: 1 if 'piwnica' in x else 0)
df['telewizor'] = df["features"].map(lambda x: 1 if 'telewizor' in x else 0)
df['monitoring'] = df["features"].map(lambda x: 1 if 'monitoring / ochrona' in x else 0)
df['oddzielna_kuchnia'] = df["features"].map(lambda x: 1 if 'oddzielna kuchnia' in x else 0)
df['garaz'] = df["features"].map(lambda x: 1 if 'garaż/miejsce parkingowe' in x else 0)
df['teren_zamkniety'] = df["features"].map(lambda x: 1 if 'teren zamknięty' in x else 0)
df['klimatyzacja'] = df["features"].map(lambda x: 1 if 'klimatyzacja' in x else 0)
df['taras'] = df["features"].map(lambda x: 1 if 'taras' in x else 0)

# create a variable that indicates price per 1sq meter
df['ppa'] = df.total_price / df.area
df['ppa'] = df['ppa'].dropna().apply(lambda x: round(x))

# take top 40 cities
top_cities = df.groupby(df.city).count().sort_values(by='rooms', ascending=False).head(40).index.values.tolist()
top_cities = sorted(top_cities)
# create new dataframe with only top 40 cities
df_clean = df[df.city.isin(top_cities)]
df_clean.drop('features', axis=1, inplace=True)

'''
################ REMOVING OUTLIERS ###################
'''

print("99% offers have area smaller than {0: .2f}".format(np.percentile(df_clean.area.dropna(), 99)))
print("99% offers have price smaller than {0: .2f}".format(np.percentile(df_clean.total_price.dropna(), 99)))

df_clean = df_clean[(df.area >= np.percentile(df_clean.area.dropna(), 1))
                    & (df.area <= np.percentile(df_clean.area.dropna(), 99))]
df_clean = df_clean[(df.total_price >= np.percentile(df_clean.total_price.dropna(), 1))
                    & (df.total_price <= np.percentile(df_clean.total_price.dropna(), 99))]
df_clean = df_clean[(df.year >= 1900) & (df.year <= 2020)]


# within a group of each city get percentiles of 'price per m2'

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


# create dataframe with 15th and 95th percentile in each city
city_percentiles = df_clean.groupby('city')['ppa'].agg([percentile(15), percentile(95), 'min', 'max'])

# exclude outliers of price per 1m2
df_clean = ps.sqldf('select a.* \
         from df_clean a join city_percentiles b \
             on a.city=b.city \
         where a.ppa between b.percentile_15 and b.percentile_95')

# drop variables having direct effect on a price
df_clean.drop('price', axis=1, inplace=True)
df_clean.drop('added_rent', axis=1, inplace=True)
df_clean.drop('ppa', axis=1, inplace=True)
df_clean.drop('private_offer', axis=1, inplace=True)

# df_clean.to_csv('otodom_cleaned.csv', index=False)
'''
################ MODEL BUILDING ###################
'''
# separate dependent and independent variables + split to training and testing dataframes
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_clean.drop('total_price', axis=1),
                                                                    df_clean.total_price, test_size=0.2, random_state=0)
print('Train size:', X_train.shape)
print('Test size:', X_test.shape)

# encode city names into numeric values
le = preprocessing.LabelEncoder()
le.fit(top_cities)
X_train.city = le.transform(X_train.city)
X_test.city = le.transform(X_test.city)

# build random forest model
clf = RandomForestRegressor(random_state=0)
clf.fit(X_train, Y_train)

# make predictions using the testing set
y_prediction = clf.predict(X_test)

# model performance
# mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, y_prediction))
# coefficient of determination:
print('Coefficient of determination: %.2f'
      % r2_score(Y_test, y_prediction))

plt.scatter(Y_test, y_prediction,  color='black')

# feature importance
feature_list = list(X_train.columns)
feature_imp = pd.Series(clf.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

print(X_train.columns)

# y_pred_2 = (pd.Series(y_pred)/100).apply(np.round).astype(int) *100

# SAVE THE MODEL
pickle.dump(clf, open('otodom_clf.pkl', 'wb'))
