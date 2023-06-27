import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV 
from sklearn.metrics import mean_squared_error, r2_score
import eli5


# ----- Creador de mapas -----

def main():
    # Configuración de la página
    st.set_page_config(page_title="NYC Airbnb")

    ## Cargamos el fichero de datos y lo almacenamos en caché
    @st.cache_data
    def load_data():
        return pd.read_csv(r"AB_NYC_2019.csv")

    # Pretratamiento del fichero de datos
    df = load_data()

    # Crear un widget de selección para las secciones
    with st.sidebar:
        st.header("Secciones")
        pages = ("Airbnb en NYC", "Precios y habitaciones en NYC", "Smart pricing")
        selected_page = st.selectbox(
            label="Elige la sección que deseas visualizar:",
            options=pages)

    ### ---- Airbnb en NYC ----

    if selected_page == "Airbnb en NYC":
        st.header("Distribución de los alquileres en NYC")
        st.subheader("Distribución de viviendas por barrios")
        st.write(
            "En este gráfico vemos representadas las diferentes viviendas disponibles en Airbnb Nueva York. El color hace referencia al barrio en donde se situan.")

        # Mapa de las viviendas por barrios
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x='longitude', y='latitude',
                        hue='neighbourhood_group', s=20, data=df)
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.title('Distribución Airbnb NYC')
        plt.legend(title="Agrupaciones de Barrios")
        st.pyplot(plt, clear_figure=True)

        # Agregamos viviendas por barrio
        st.subheader("Número de viviendas por barrio")
        neight_count = df.groupby('neighbourhood_group').agg('count').reset_index()
        cantidades = {elem[0]: elem[1] for elem in neight_count.values}
        barrios = list(cantidades.keys())
        hood = st.selectbox('Selecciona un barrio:', barrios)
        if hood == barrios[0]:
            st.write(f'El número de viviendas en {barrios[0]} es de {cantidades[barrios[0]]}')
        elif hood == barrios[1]:
            st.write(f'El número de viviendas en {barrios[1]} es de {cantidades[barrios[1]]}')
        elif hood == barrios[2]:
            st.write(f'El número de viviendas en {barrios[2]} es de {cantidades[barrios[2]]}')
        elif hood == barrios[3]:
            st.write(f'El número de viviendas en {barrios[3]} es de {cantidades[barrios[3]]}')
        else:
            st.write(f'El número de viviendas en {barrios[4]} es de {cantidades[barrios[4]]}')

        # Usamos geopandas para construir una capa base de los barrios de NYC
        st.subheader("Representación de los vecindarios con geopandas")
        st.write(
            "Con el uso de geopandas, podemos obtener el área de cada uno de los barrios. El código es el siguiente:")
        st.code("gpd.read_file(gpd.datasets.get_path('nybb'))")
        nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
        st.write("Que convertido a dataframe queda de la siguiente manera:")

        df_neight = pd.DataFrame(nyc.drop('geometry', axis=1))
        st.dataframe(df_neight.head(5))

        nyc.rename(columns={'BoroName': 'neighbourhood_group'}, inplace=True)
        bc_geo = nyc.merge(neight_count,
                           on='neighbourhood_group')  # DataFrame agregado con el número de viviendas por barrio y su área

        # Creamos el mapa
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        bc_geo.plot(column='id', cmap='viridis_r', alpha=.5, ax=ax1, legend=True)
        bc_geo.apply(lambda x: ax1.annotate(text=x.neighbourhood_group, color='black', xy=x.geometry.centroid.coords[0],
                                            ha='center'), axis=1)
        plt.title("Número de alquileres por barrio en NYC")
        plt.axis('off')
        st.pyplot(fig1, clear_figure=True)

    ### ---- Precios en NYC ----

    if selected_page == "Precios y habitaciones en NYC":
        st.header("Análisis de los precios y tipos de habitación")
        st.subheader("Densidad y distribución de los precios por barrio")

        # Almacenamos la información sobre cada uno de los barrios

        # Brooklyn
        sub_1 = df.loc[df['neighbourhood_group'] == 'Brooklyn']
        price_sub1 = sub_1[['price']]
        # Manhattan
        sub_2 = df.loc[df['neighbourhood_group'] == 'Manhattan']
        price_sub2 = sub_2[['price']]
        # Queens
        sub_3 = df.loc[df['neighbourhood_group'] == 'Queens']
        price_sub3 = sub_3[['price']]
        # Staten Island
        sub_4 = df.loc[df['neighbourhood_group'] == 'Staten Island']
        price_sub4 = sub_4[['price']]
        # Bronx
        sub_5 = df.loc[df['neighbourhood_group'] == 'Bronx']
        price_sub5 = sub_5[['price']]
        # Ponemos todos los df en una lista
        price_list_by_n = [price_sub1, price_sub2, price_sub3, price_sub4, price_sub5]

        # creating an empty list that we will append later with price distributions for each neighbourhood_group
        p_l_b_n_2 = []
        # creating list with known values in neighbourhood_group column
        nei_list = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
        # creating a for loop to get statistics for price ranges and append it to our empty list
        for x in price_list_by_n:
            i = x.describe(percentiles=[.25, .50, .75])
            i = i.iloc[3:]
            i.reset_index(inplace=True)
            i.rename(columns={'index': 'Stats'}, inplace=True)
            p_l_b_n_2.append(i)
        # changing names of the price column to the area name for easier reading of the table
        p_l_b_n_2[0].rename(columns={'price': nei_list[0]}, inplace=True)
        p_l_b_n_2[1].rename(columns={'price': nei_list[1]}, inplace=True)
        p_l_b_n_2[2].rename(columns={'price': nei_list[2]}, inplace=True)
        p_l_b_n_2[3].rename(columns={'price': nei_list[3]}, inplace=True)
        p_l_b_n_2[4].rename(columns={'price': nei_list[4]}, inplace=True)
        # finilizing our dataframe for final view
        stat_df = p_l_b_n_2
        stat_df = [df.set_index('Stats') for df in stat_df]
        stat_df = stat_df[0].join(stat_df[1:])
        st.write(
            "Como podemos observar a continuación, los valores máximos de los precios para cada uno de los barrios son muy altos. Por tanto, vamos a establecer un límite de 500$ para poder realizar un mejor entendimiento y representación.")
        st.dataframe(stat_df)

        # Creación del violinplot

        # creating a sub-dataframe with no extreme values / less than 500
        sub_6 = df[df.price < 500]
        # using violinplot to showcase density and distribtuion of prices
        viz_2 = sns.violinplot(data=sub_6, x='neighbourhood_group', y='price')
        viz_2.set_title('Densidad y distribución de los precios para cada barrio')
        viz_2.set_xlabel('Nombre del barrio')
        viz_2.set_ylabel('Precio en $')
        st.pyplot(plt.gcf())  # se utiliza plt.gcf() para obtener la figura actual
        st.write(
            "Con la tabla estadística y el gráfico de violín podemos observar algunas cosas sobre la distribución de precios de Airbnb en los distritos de la ciudad de Nueva York. En primer lugar, podemos afirmar que Manhattan tiene el rango de precios más alto para las publicaciones, con un precio promedio de 150 dólares por noche, seguido de Brooklyn con 90 dólares por noche. Queens y Staten Island parecen tener distribuciones muy similares, mientras que Bronx es el más económico de todos. Esta distribución y densidad de precios eran totalmente esperadas; por ejemplo, no es un secreto que Manhattan es uno de los lugares más caros del mundo para vivir, mientras que Bronx, por otro lado, parece tener un nivel de vida más bajo.")

        # Tipo de habitación

        st.subheader("Tipos de habitación por distrito")
        hood1 = st.selectbox("Selecciona el barrio que deseas visualizar:", nei_list + ["Todos"])
        agregado_price = sub_6.groupby(['neighbourhood_group', 'room_type']).agg({'price': 'mean'})
        agregado_price1 = agregado_price[agregado_price.price < 500]
        agregado_price1 = agregado_price1.reset_index()
        if hood1 != "Todos":
            sub_7 = df.loc[df["neighbourhood_group"] == hood1]
            viz_3 = sns.catplot(x='neighbourhood_group', col='room_type', data=sub_7, kind='count')
            viz_3.set_xlabels('')
            viz_3.set_ylabels('Nº de habitaciones')
            viz_3.set_xticklabels(rotation=90)
            st.pyplot(viz_3)
            st.write(f"Los precios promedios para cada tipo de habitación en el distrito {hood1} son:")
            st.dataframe(agregado_price1.loc[agregado_price1["neighbourhood_group"] == hood1])
            st.write(
                "Ten en cuenta que este promedio es teniendo en cuenta solo aquellos alquileres cuyo precio es inferior 500 dólares.")
        else:
            st.pyplot(sns.catplot(x='neighbourhood_group', hue='neighbourhood_group', col='room_type', data=sub_6,
                                  kind="count"))
            st.write("Estos son los precios promedio para cada habitación por barrio:")
            st.dataframe(agregado_price)
            st.write(
                "Ten en cuenta que este promedio es teniendo en cuenta solo aquellos alquileres cuyo precio es inferior a 500 dólares.")

    ### ---- Predicción del precio ----

    if selected_page == "Smart pricing":
        df = df[["neighbourhood_group", "neighbourhood", "latitude", "longitude", "room_type", "price", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"]]
        st.header("Predicción del precio de un alquiler basado en las características de la vivienda")
        st.subheader("Distribución del precio")
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        sns.distplot(df['price'], ax=axes[0])
        axes[0].set_xlabel("Precio")
        sns.distplot(np.log1p(df['price']), ax=axes[1])
        axes[1].set_xlabel('log(1+price)')
        sm.qqplot(np.log1p(df['price']), stats.norm, fit=True, line='45', ax=axes[2]);
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        st.write(
            "Como podemos observar en el primer gráfico, la distribución del precio de asemeja mucho a una distribución log-normal. Por tanto, aplicamos una transformación logarítmica para obtener una distribución lo más cercana a una normal.")
        st.write(
            "En el último gráfico el eje x del gráfico representa los cuantiles teóricos de una distribución normal, mientras que el eje y representa los cuantiles de los datos observados.")
        st.write(
            "Además, vamos a acotar los valores de manera que dejemos fuera los outliers. En este caso, basándonos en el gráfico central, vamos a quedarnos únicamente con aquellos valores que se situen entre el 3 y el 8.")
        
        # Filtramos
        st.write("Este es el resultado después del filtrado:")
        df = df[np.log1p(df['price']) < 8] 
        df = df[np.log1p(df['price']) > 3] 
        fig1, axes1 = plt.subplots(1, 3, figsize=(21, 6))
        sns.distplot(df['price'], ax=axes1[0])
        axes1[0].set_xlabel("Precio")
        sns.distplot(np.log1p(df['price']), ax=axes1[1])
        axes1[1].set_xlabel('log(1+price)')
        sm.qqplot(np.log1p(df['price']), stats.norm, fit=True, line='45', ax=axes1[2]);
        st.pyplot(fig1)

        st.subheader("Preprocesamiento de los datos")

        # Tomando una fracción de los datos para evitar problemas de memoria
        sample_fraction = 0.1  # Ajusta este valor según la capacidad de tu máquina
        df = df.sample(frac=sample_fraction, random_state=1)

        # Encoding de características categóricas
        categorical_features = df.select_dtypes(include=['object'])
        categorical_features_one_hot = pd.get_dummies(categorical_features)

        # Rellenar los valores nulos
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

        # Seleccionar características numéricas
        numerical_features = df.select_dtypes(exclude=['object'])
        y = np.log1p(numerical_features.price)  # Hemos aplicado log-transform a y
        numerical_features = numerical_features.drop(['price'], axis=1)

        # Concatenar características numéricas y categóricas codificadas
        X = pd.concat([numerical_features, categorical_features_one_hot], axis=1)

        st.write("Tomando una muestra aleatoria del 10% de los datos para evitar problemas de memoria, Realizando One-Hot Encoding en las características categóricas, Concatenando características numéricas y categóricas codificadas en un solo dataframe, Dividiendo el dataset en conjuntos de entrenamiento y prueba,  Reescalando las características usando RobustScaler.")

        # División entre conjuntos de entrenamiento y prueba
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reescalado de las características
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.subheader("Entrenar un modelo con Random Forest")
        
        
        # Initialize RF model
        
        st.write("Estos son los resultados del modelo entreando: ")
        
        st.write("\n")
        
        rfr_baseline = RandomForestRegressor(random_state=42)
        
        rfr_baseline.fit(X_train, y_train) 
        y_train_rfr = rfr_baseline.predict(X_train)
        y_test_rfr = rfr_baseline.predict(X_test)
        
        mse_train = mean_squared_error(y_train, y_train_rfr)
        st.write("Error cuadrático medio (MSE) en el conjunto de entrenamiento:", mse_train)
        
        # Calcular el coeficiente de determinación en el conjunto de entrenamiento
        r2_train = r2_score(y_train, y_train_rfr)
        st.write("Coeficiente de determinación (R^2) en el conjunto de entrenamiento:", r2_train)
        
        # Calcular el error cuadrático medio en el conjunto de prueba
        mse_test = mean_squared_error(y_test, y_test_rfr)
        st.write("Error cuadrático medio (MSE) en el conjunto de prueba:", mse_test)
        
        # Calcular el coeficiente de determinación en el conjunto de prueba
        r2_test = r2_score(y_test, y_test_rfr)
        st.write("Coeficiente de determinación (R^2) en el conjunto de prueba:", r2_test)
        
        st.write("\n")
        weights = eli5.format_as_dataframe(eli5.explain_weights(rfr_baseline, feature_names=list(X.columns)))

        # Crear un expander en Streamlit
        with st.expander("Mostrar pesos de características"):
            st.dataframe(weights)
        
        st.subheader("Averigua el precio recomendado para tu vivienda")
        st.write("Introduce las características:")
        st.write("\n")
        neighbourhood_group = st.selectbox("Elige el distrito en el que se situa:", list(df["neighbourhood_group"].unique()))
        neighbourhood = st.selectbox("Elige el barrio en el que se situa:", sorted(list(df["neighbourhood"].unique())))
        latitude = st.slider('Selecciona la latitud:', min_value=40.49979, max_value=40.91306, step=0.00001)
        longitude = st.slider('Selecciona la longitud:', min_value=-74.24442, max_value=-73.71299, step=0.00001)
        room_type = st.selectbox("Selecciona el tipo de habitación", list(df["room_type"].unique()))
        minimum_nights = st.slider("Selecciona el nº de noches mínimo:", min_value=int(min(df["minimum_nights"].unique())), max_value=int(max(df["minimum_nights"].unique())))
        number_of_reviews = st.slider("Selecciona el nº de reviews:", min_value=int(min(df["number_of_reviews"].unique())), max_value=int(max(df["number_of_reviews"].unique())))
        reviews_per_month = st.slider("Reviews por mes:", min_value = 0, max_value = 60)
        calculated_host_listings_count = st.slider("Cuántas casas tienes ofertadas:", min_value = 1, max_value = int(max(df["calculated_host_listings_count"].unique())))
        availability_365 = st.slider("Días disponibles durante el año:", min_value = int(min(df["availability_365"].unique())), max_value = int(max(df["availability_365"].unique())))
         
        # Una vez seleccionadas las características, calculamos la predicción.
        aux = list(X.iloc[0]) # Lista auxiliar para crear la nueva instancia
        aux1 = [False for i in range(len(aux))]
        
        columns = ['neighbourhood_group'+f'_{neighbourhood_group}', 'neighbourhood'+f'_{neighbourhood}', 'latitude', 'longitude',
       'room_type'+f'_{room_type}', 'minimum_nights', 'number_of_reviews',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365']
        
        var = [neighbourhood_group, neighbourhood, latitude, longitude,
                        room_type, minimum_nights, number_of_reviews,
                        reviews_per_month, calculated_host_listings_count,
                        availability_365]

        col_index = []
        for i in range(len(columns)): 
            index = X.columns.get_loc(columns[i])
            res = var[i]
            if isinstance(res, str):
                aux1[index] = 1.0
            else: 
                aux1[index] = var[i]        
        
        aux1 = np.array(aux1)
        new = scaler.transform(aux1.reshape(1, -1))  
        res_new = rfr_baseline.predict(new.reshape(1, -1)) 

        media = df["price"].mean()
        dv = df["price"].std()
        new_desescalado = (res_new * dv) + media
        st.write("Este es el precio obtenido en dólares:", new_desescalado[0])

if __name__ == "__main__":
    main()







