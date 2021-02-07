import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# Set up page layout
st.set_page_config(page_title='Wyceń swoje mieszkanie', page_icon='🏡', layout='wide', initial_sidebar_state='auto')

# Read data
df_raw = pd.read_csv('otodom_cleaned.csv')

st.write('''
# Wyceń swoje mieszkanie!

Ta aplikacja pozwala dokonać **wyceny mieszkania** na podstawie zdefiniowanych paramtetrów.
Model predykcyjny został stworzony na podstawie danych z [`www.otodom.pl`](#https://www.otodom.pl).
''')

st.sidebar.header('Wybierz parametry:')

cities = list(sorted(set(df_raw.city)))


def user_input_features():
    city = st.sidebar.selectbox('Miasto', cities, 34)
    rooms = st.sidebar.slider('Liczba pokoi', 1, 10, 2)
    area = st.sidebar.number_input('Powierzchnia [m2]', 0, 200, 32)
    private_offer = st.sidebar.radio("Oferta dodana przez:", ("Osoba prywatna", 'Agencja nieruchomości'))
    year = st.sidebar.slider('Rok budowy', 1900, 2021, 2016)
    zmywarka = st.sidebar.checkbox('Zmywarka')
    telewizor = st.sidebar.checkbox('Telewizor')
    klimatyzacja = st.sidebar.checkbox('Klimatyzacja')
    oddzielna_kuchnia = st.sidebar.checkbox('Oddzielna kuchnia')
    balkon = st.sidebar.checkbox('Balkon')
    taras = st.sidebar.checkbox('Taras')
    piwnica = st.sidebar.checkbox('Piwnica')
    garaz = st.sidebar.checkbox('Garaż')
    monitoring = st.sidebar.checkbox('Monitoring')
    teren_zamkniety = st.sidebar.checkbox('Teren zamknięty')


    data = {'rooms': rooms,
            'area': int(area),
            'city': city,
            'private_offer': 1 if private_offer=='Osoba prywatna' else 0,
            'year': year,
            'zmywarka': zmywarka,
            'balkon': balkon,
            'piwnica': piwnica,
            'telewizor': telewizor,
            'monitoring': monitoring,
            'oddzielna_kuchnia': oddzielna_kuchnia,
            'garaz': garaz,
            'teren_zamkniety': teren_zamkniety,
            'klimatyzacja': klimatyzacja,
            'taras': taras
            }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
df = df_raw.drop(columns=['total_price'])
df = pd.concat([input_df, df], axis=0)

# Encoding of ordinal features
le = preprocessing.LabelEncoder()
le.fit(cities)
df.city = le.transform(df.city)
df = df[:1]  # Selects only the first row (the user input data)

# Create nicer df for display
df_pretty = df.rename(columns={'rooms': 'Liczba pokoi',
                               'area': 'Powierzchnia [m2]',
                               'city': 'Miasto',
                               'private_offer': 'Oferta prywatna',
                               'year': 'Rok budowy',
                               'zmywarka': 'Zmywarka',
                               'balkon': 'Balkon',
                               'piwnica': 'Piwnica',
                               'telewizor': 'Telewizor',
                               'monitoring': 'Monitoring',
                               'oddzielna_kuchnia': 'Oddzielna kuchnia',
                               'garaz': 'Garaż',
                               'teren_zamkniety': 'Teren zamknięty',
                               'klimatyzacja': 'Klimatyzacja',
                               'taras': 'Taras'
                               })

# Display name of a city -> inverse label encoding
df_pretty['Miasto'] = le.inverse_transform(df_pretty['Miasto'])

# Hide index in the table
blankIndex = [''] * len(df_pretty)
df_pretty.index = blankIndex

# Displays the user input features
st.subheader('Parametry wycenianego mieszkania: ')
st.table(df_pretty.iloc[:, :8])
st.table(df_pretty.iloc[:, 8:])
# Reads in saved classification model
load_clf = pickle.load(open('otodom_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

st.subheader('Szacowana cena wybranego mieszkania wynosi: ')
prediction_r = (pd.Series(prediction) / 100).apply(np.round).astype(int) * 100
st.markdown(f'**{prediction_r[0]}**')
