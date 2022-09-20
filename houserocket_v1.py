# Imported Libs
import streamlit as st
import pandas as pd
import numpy as np
import folium
import geopandas
from datetime import datetime
import plotly.express as px
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def load_data(path):
    dataframe = pd.read_csv(path)
    return dataframe

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def set_features(data):
    data['price_m2'] = data['price'] / (data['sqft_lot'] / 0.092903)
    return data


def overview_data (data):
    # In App
    st.title('House Rocket Dataset')
    st.subheader('This is a initial project about the dataset House Rocket.')
    st.write('version 1.0')

    f_attributes = st.multiselect('Enter Columns', data.columns)
    f_zipcode = st.multiselect('Enter Zipcode', data.zipcode.unique())

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]

    else:
        data = data.copy()

    # New features

    # Average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge DFs
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL_HOUSES', 'PRICE', 'SQFT_LIVING', 'PRICE/M²']

    num_attributes = data.select_dtypes('int64', 'float64')
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    dfn = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    dfn.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

    col1, col2 = st.columns(2)

    col1.dataframe(df)
    col2.dataframe(dfn)

    return None

def portfolio_density(data, geofile):
    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    # Base Map
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=10)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in data.iterrows():
        folium.Marker([row['lat'], row['long']], popup="Sold R$ {0} on: {1}. Features {2} sqft, {3} bedrooms, "
                                                       "{4} bathrooms, year built: {5}".format(row['price'],
                                                                                               row['date'],
                                                                                               row['sqft_living'],
                                                                                               row['bedrooms'],
                                                                                               row['bathrooms'],
                                                                                               row['yr_built'])).add_to(
            marker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map

    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    region_price_map.choropleth(data=df, geo_data=geofile, columns=['ZIP', 'PRICE'], key_on='feature.properties.ZIP',
                                fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2, legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)

    return None

def comertial(data):
    st.title('Commercial Options')
    st.title('Commercial Attributes')

    # --------------Average Price Per Year

    # Filters

    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.subheader('Select Max Year Built')

    f_year_built = st.slider('Year Built:', min_year_built, max_year_built, min_year_built)

    st.header('Average Price per Year Built')

    dg1 = data.loc[data['yr_built'] < f_year_built]
    tab1 = dg1[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig = px.line(tab1, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # --------------Average Price Per Day
    data['date'] = pd.to_datetime(data['date']).dt.strftime('Y%-m%-d%')

    st.header('Average Price per Day)')
    st.subheader('Select Max Date')

    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d %H:%M:%S')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d %H:%M:%S')

    f_date = st.slider('Date', min_date, max_date, min_date)

    data['date'] = pd.to_datetime(data['date'])
    dg2 = data.loc[data['date'] < f_date]
    tab2 = dg2[['date', 'price']].groupby('date').mean().reset_index()

    fig = px.line(tab2, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ==================== Histogram

    st.header('Price Distribution')
    st.subheader('Select Max Price')

    # filter

    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.slider('Price', min_price, max_price, avg_price)

    dfp = data.loc[data['price'] < f_price]

    fig = px.histogram(dfp, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    # Distribuições dos Imóveis por categorias físicas

    st.title('Attributes Options')
    st.title('House Attributes')

    f_bedrooms = st.selectbox('Max Number of Bedrooms', data['bedrooms'].sort_values().unique())
    f_bathrooms = st.selectbox('Max Number of Bedrooms', data['bathrooms'].sort_values().unique())

    c1, c2 = st.columns(2)

    # houses per bedrooms
    c1.header('Houses per Bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=15)
    c1.plotly_chart(fig, use_container_width=True)

    # houses per bathrooms
    c2.header('Houses per Bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=15)
    c2.plotly_chart(fig, use_container_width=True)

    f_floors = st.selectbox('Max number of floor', data['floors'].sort_values().unique())
    f_waterview = st.checkbox('Only houses with waterview', data['waterfront'].unique().all())

    c1, c2 = st.columns(2)

    # houses per floor
    c1.header('Houses per Floor')
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=15)
    c1.plotly_chart(fig, use_container_width=True)

    # houses per waterfront
    c2.header('Houses per Waterfront')
    if f_waterview:
        df = data[data['waterfront']==1]

    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=15)
    c2.plotly_chart(fig, use_container_width=True)

    return None

if __name__ == '__main__':

    # Data Extraction

    # Get Data
    path = 'kc_house_data.csv'
    data = load_data(path)

    # Get Geofile
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    geofile = get_geofile(url)

    # Transformation
    data = set_features(data)

    overview_data(data)

    portfolio_density(data, geofile)

    comertial(data)

