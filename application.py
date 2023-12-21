import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
from streamlit_option_menu import option_menu
from social_media import *
import time
from custom import custom_css
import joblib


warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Dimensionless influx of a reseviors", page_icon=":histogram:")


def show_data():
    pass


def main():
    linkedin_url = "https://linkedin.com/in/emmanuel-esin-957834200"
    twitter_url = "https://twitter.com/esin_professor"
    #linkedin_icon_url = "images/linkedin_icon.png"
    #twitter_icon_url = "images/twitter_icon.png"
    personal_image_url = "C:/Users/HP/Downloads/0EJzHDnp_400x400.jpg"

    with st.sidebar:
        # Add a title with an icon
        st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">', unsafe_allow_html=True)
        st.markdown('''
                    <h1 style='text-align: center;'>
                        <i class='fas fa-star'></i> Menu <i class='fas fa-home'></i>
                    </h1>
                    ''', unsafe_allow_html=True)
        options = st.sidebar.selectbox("Select Option", ["Home", "Modelling", "Evaluation"])

    if options == "Home":
        st.markdown(
            """<h3 style="text-align: center"> Prediction of Dimensionless Water Influx for Water Influx estimation in reservoirs using Machine Learning </h3>""",
            unsafe_allow_html=True
            )

        marquee_text = "A Final Year project done by EMMNAUEL WILLIAM ESIN "

        # HTML and CSS for the marquee
        marquee_html = f"""
            <style>
                .marquee {{
                    white-space: nowrap;
                    overflow: hidden;
                    padding: 10px;
                }}
                .marquee p {{
                    display: inline-block;
                    animation: marquee 15s linear infinite;
                }}
                @keyframes marquee {{
                    0% {{ transform: translateX(100%); }}
                    100% {{ transform: translateX(-100%); }}
                }}
            </style>
            <div class="marquee">
                <p>{marquee_text}</p>
            </div>
        """

        # Display the marquee
        st.markdown(marquee_html, unsafe_allow_html=True)
        st.write("This app allows you to upload and analyze your water influx data, train with various machine learning models, generate predictions and carry out some other evaluations")
        
        columy2, coly = st.columns(2)
        
        with columy2:
            
            st.write("**Key features:**")
            st.write("* Upload and visualize your water influx data.")
            st.write("* Choose from different machine learning models for prediction.")
            st.write("* Tune model hyperparameters for optimal performance.")
            st.write("* Analyze model performance using various metrics.")
            st.write("* Generate and download predictions based on your data.")
            st.write("* Explore relevant resources and documentation.")
            # local_image_path = "../Emmyesin/gold.jpg"
            
            with st.sidebar:
                photo_url = "https://uploads-ssl.webflow.com/61eeba8765031c95bb83b2ea/61fbec562cf81f62a255f192_61eeb99a54a67e18ce19d47c_0_nyBFE8lLgr8ePAJ_%20(1).jpeg"
                st.markdown(
                f"""
                    <div style='display: flex; justify-content: center; align-items: center; height: 250px; width: 100%; border-radius: 70%; overflow: hidden; background-color: #f0f0f0;'>
                        <img src="{photo_url}" style='width: 100%; height: 100%; object-fit: cover; border-radius: 50%;'>
                    </div>

                    <h2 style="text-align: center"> Machine Learning </h2>
                    <p> 
                        Within the discipline of artificial intelligence (AI), machine learning is the study of creating statistical models and algorithms that allow computer systems to learn from data and perform better on a given task without the need for explicit programming.
                        Put differently, machine learning refers to the development of systems that can learn and develop on their own through experience or examples, enabling them to classify data, make predictions, or carry out tasks without explicit guidance.
                    </p>
                    """,
                    unsafe_allow_html=True
            )
                st.markdown("Follow me on:")
                social_media_buttons()
            
        with coly:
            st.markdown(custom_css, unsafe_allow_html=True)
            image_urls = [
                "https://www.researchgate.net/profile/Yassine-Al-Amrani/publication/323843520/figure/fig1/AS:806483344248835@1569292021708/Sentiment-analysis-model.jpg",
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQbB7NRALylccKZzYTVUM5XXxK5105FcSamdA&usqp=CAU",
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTK2ssbq78u2yHD6vOkiWzx_Aq0y7fSJWjT5A&usqp=CAU",
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8mmjYxXzDI4C5FCYrabbwMveRsr-Aby_31Q&usqp=CAU"
            ]

            carousel_html = '<div class="carousel-container">'
            for image_url in image_urls:
                carousel_html += f'<div class="carousel-item"><img src="{image_url}" alt="Image" style="margin : 20px" class="animate"></div>'
            carousel_html += "</div>"
            st.markdown(carousel_html, unsafe_allow_html=True)

        #st.image("images/linkedin_icon.png", width=32, height=32)
        #st.write("https://www.linkedin.com/in/emmanuel-esin-957834200/")
        #st.image("images/twitter_icon.png", width=32, height=32)
        #st.write("https://twitter.com/esin_professor")


    # elif options == "Visualizations":
    #     st.title("Data Visualizations")

    #     uploader_file = st.file_uploader("Upload the datasets for analysis", type=['csv'])

    #     if uploader_file is not None:
    #         @st.cache_data
    #         def get_cdv_data():
    #             df = pd.read_csv(uploader_file)
    #             return df

    #         df = get_cdv_data()

    #         col1, col2 = st.columns(2)

    #         with col1:
    #             st.write("Original Data")
    #             search_query = st.text_input("Search by value:", "")

    #             # Filter the DataFrame based on the search query
    #             df_filtered = filter_data(df[["tD", "WeD"]], search_query)

    #             # Display the table with a fixed number of rows and a vertical scrollbar
    #             st.markdown(
    #                 f"""
    #                 <div style="overflow-y: scroll; max-height: 400px; width: 500px">
    #                     {df_filtered.to_html(classes="dataframe")}
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True
    #             )
    #         with col2:
    #             st.set_option('deprecation.showPyplotGlobalUse', False)
    #             st.title("Visualizing the Entire Data")
    #             plt.title("A graph of tD Vs WeD")
    #             plt.xlabel("tD(Time Dimension")
    #             plt.ylabel("WeD(Water influx)")
    #             plt.scatter(df['tD'], df['WeD'])
    #             st.pyplot()

    elif options == "Modelling":
        st.title("Data Training and Prediction")
        with st.sidebar:
            option = option_menu("Step 1", ["Importing data"], icons=['home'])

        uploader_file = st.file_uploader("Upload the datasets for analysis", type=['csv'])
        
        

        if uploader_file is not None:
            @st.cache_data
            def get_cdv_data():
                df = pd.read_csv(uploader_file)
                return df

            
            html = """<h2 style='text-align: center'> Previewing Datasets and Visualization </h2>"""
            st.write(html, unsafe_allow_html=True)
            with st.spinner("loading..."):
                time.sleep(2)
            df = get_cdv_data()

            col1, col2 = st.columns(2)

            with col1:
                st.write("Original Data")

                # frame = {'size': ['rows', 'cols'], "data": [df.shape[0], df.shape[1]]}
                # st.write(frame['size'], frame['data'])
                search_query = st.text_input("Search by value:", "")

                # Filter the DataFrame based on the search query
                df_filtered = filter_data(df[["tD", "WeD"]], search_query)

                # Set the maximum number of rows to display initially
                max_rows_to_display = 11

                # Display the table with a fixed number of rows and a vertical scrollbar
                st.markdown(
                    f"""
                        <div style="overflow-y: scroll; max-height: 400px; width: 500px">
                            {df_filtered.to_html(classes="dataframe")}
                        </div>
                        """,
                    unsafe_allow_html=True
                )

                # st.write(df_filtered)
            with col2:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.title("Visualizing the Entire Data")
                plt.title("A graph of tD Vs WeD")
                plt.xlabel("tD(Time Dimension")
                plt.ylabel("WeD(Water influx)")
                plt.scatter(df['tD'], df['WeD'])
                st.pyplot()

            st.title("Model Training")
            with st.sidebar:
                    option_menu("Step 2", ["Model Training"], icons=['about'])

            # data splitting
            value = st.number_input("Enter a length of the data to analyse", key="stream", min_value=0.0,
                                    max_value=1000000.0, step=0.01)

            with st.spinner("Loading........."):
                time.sleep(0.5)
            
            
            cola, colb = st.columns(2)
            if value < 20:
                st.write("can\'t work with a value less then 20")
            else:
                
                x = df['tD'][:int(value)]
                y = df['WeD'][:int(value)]

                
                with cola:
                    st.write("Visualizing the Datasets")
                    plt.figure(figsize=(8, 4))
                    plt.scatter(x, y, color="red", marker="o")
                    plt.title("A graph of tD against WeD")
                    plt.xlabel("tD")
                    plt.ylabel("WeD")
                    plt.grid()
                    # plt.plot(x,y)
                    st.pyplot()

                with colb:
                    st.write("Visualizing the Training Datasets")
                    # training the dataset
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
                    plt.figure(figsize=(8, 4))
                    plt.scatter(x_train, y_train, color="green", marker="o")
                    plt.title("A graph of trained tD against trained WeD")
                    plt.xlabel("tD")
                    plt.ylabel("WeD")
                    plt.grid()
                    # plt.plot(x,y)
                    st.pyplot()

                # reshaping datasets
                x_train_reshaped = np.array(x_train).reshape(-1, 1)
                x_test_reshaped = np.array(x_test).reshape(-1, 1)

                y_train_reshaped = np.array(y_train).reshape(-1, 1)
                y_test_reshaped = np.array(y_test).reshape(-1, 1)

                # Modelling/ Algorithms analyses
                st.title("Model Analysis")
                with st.sidebar:
                    option_menu("Step 3", ["Model Analysis"], icons=["menu"])
                
                # Create a sidebar for selection
                colum1, colum2, colum3 = st.columns(3)
                
                with colum1:
                    st.text("XgBoost")
                    
                with colum2:
                    st.text("Linear Regreesion")
                    
                with colum3:
                    st.text("Random Forest Regression")

                
                lr = LinearRegression()
                lr.fit(x_train_reshaped, y_train_reshaped)
                y_pred1 = lr.predict(x_test_reshaped)

                # using Random-forest
                rf = RandomForestRegressor()
                rf.fit(x_train_reshaped, y_train_reshaped)
                y_pred2 = rf.predict(x_test_reshaped)

                # using XGBoost
                xg = XGBRegressor()
                xg.fit(x_train_reshaped, y_train_reshaped)
                y_pred3 = xg.predict(x_test_reshaped)

                plt.figure(figsize=(10, 4))
                plt.scatter(range(len(y_test)), y_test, label='Actual Data', color="brown", marker="o")
                plt.scatter(range(len(y_test)), y_pred3, label='Predicted Values', color="grey", marker="o")
                plt.title('XGBoost Regression')
                plt.xlabel("predicted Value")
                plt.ylabel("Actual Value")
                plt.legend()
                st.pyplot()

                fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                # Plot data in the first subplot (upper left)
                axs[0].scatter(range(len(y_test)), y_test, label='Actual Values', linestyle="-", marker="o")
                axs[0].scatter(range(len(y_test)), y_pred1, label='Predicted Values', linestyle="-", marker="o")
                axs[0].set_title('Linear Regression')
                axs[0].set_xlabel("predicted Value")
                axs[0].set_ylabel("Actual Value")

                # Plot data in the second subplot (upper right)
                axs[1].scatter(range(len(y_test)), y_test, label='Actual Values', marker="o")
                axs[1].scatter(range(len(y_test)), y_pred2, label='Predicted Values', marker="o")
                axs[1].set_title('Random Forest Regression')
                axs[1].set_xlabel("predicted Value")
                axs[1].set_ylabel("Actual Value")

                plt.subplots_adjust(hspace=0.3)
                # # Add legend to all subplots
                for ax in axs.flat:
                    ax.legend()

                # Plot data in the fourth subplot (lower right)

                # Adjust layout for better spacing
                st.pyplot(fig)

                # finding the error(mean absolute error and mean squared error)
                # for linear regression
                lr_mae = mean_absolute_error(y_test, y_pred1)
                lr_rmse = mean_squared_error(y_test, y_pred1, squared=False)
                lr_ame = lr_mae / len(y_test)

                # for random-forest
                rf_mae = mean_absolute_error(y_pred2, y_test)
                rf_rmse = mean_squared_error(y_pred2, y_test, squared=False)
                rf_ame = rf_mae / len(y_test)

                # for XGBoost
                xg_mae = mean_absolute_error(y_test, y_pred3)
                xg_rmse = mean_squared_error(y_test, y_pred3, squared=False)
                xg_ame = rf_mae / len(y_test)

                # ''' displaying the errors in a table'''
                with st.sidebar:
                    option_menu("Step 4", ["Performance Analysis"], icons=["menu"])
                data = pd.DataFrame(
                    {"Model Name": ["Linear Regression", "RandomForest Regression",
                                    "XGBoost Regression"],
                     "Mean Absolute Errors": [lr_mae, rf_mae, xg_mae],
                     "Mean Squared Errors": [lr_rmse, rf_rmse, xg_rmse],
                     })

                st.title("Performance Analysis")
                left_col, right_col = st.columns(2)

                with left_col:
                    st.table(data)
                    st.write('''
                            The Mean Error and Root Mean Errors are used to ensure the accuracy of the machine
                            learning model/algorithms which is best used to predict the expected outcome of the
                            relevent output.In this instance the mean error and the root mean error has to be very 
                            little to give an approximate value of WeD.
                        ''')
                with right_col:
                    draw_graph(lr_mae, rf_mae, xg_mae, lr_rmse, rf_rmse, xg_rmse)


                st.title("Data Predictions")
                colm, colp = st.columns(2)

                with colm:
                    td = st.number_input("Enter the tD value", min_value=0.0,
                                            max_value=1000000.0, step=0.01)
                    
                    # Initialize session state
                    st.session_state.data = []
                    
                    if 'data' not in st.session_state:
                        st.session_state.data = []

                    if st.button("Predict"):
                        model_xg = xg.predict([[td]])
                        model_rf = rf.predict([[td]])
                        model_lr = lr.predict([[td]])
                        st.spinner("Predicting.......")

                        with colp:
                            new_data = pd.DataFrame(
                                {"Model Name": ["XGBoost Regression", "RandomForest Regression",
                                                "Linear Regression"],
                                    "Outcome": [model_xg[0], model_rf[0], model_lr[0]],
                                    })

                            st.write(new_data)
                            
                            
                        linear = joblib.dump(model_lr, "linear_regression")
                        random = joblib.dump(model_rf, "random_forest")
                        xgboost = joblib.dump(model_xg, "xgboost regression")
                        
                        models = pd.DataFrame(
                            {"Model Name": ["XGBoost Regression", "RandomForest Regression",
                                            "Linear Regression"],
                                "Outcome": [linear, random, xgboost],
                            })

                        st.session_state.data = models
                        st.table(st.session_state.data)
                
                    if st.session_state.data is not None:
                        st.success("Analysis Completed!!! data saved Successfully")

    else:
        with st.sidebar:
            photo_url = "https://uploads-ssl.webflow.com/61eeba8765031c95bb83b2ea/61fbec562cf81f62a255f192_61eeb99a54a67e18ce19d47c_0_nyBFE8lLgr8ePAJ_%20(1).jpeg"
            st.markdown(
                f"""
                    <div style='display: flex; justify-content: center; align-items: center; height: 250px; width: 100%; border-radius: 70%; overflow: hidden; background-color: #f0f0f0;'>
                        <img src="{photo_url}" style='width: 100%; height: 100%; object-fit: cover; border-radius: 50%;'>
                    </div>

                    <h2 style="text-align: center"> Machine Learning </h2>
                    <p> 
                        Within the discipline of artificial intelligence (AI), machine learning is the study of creating statistical models and algorithms that allow computer systems to learn from data and perform better on a given task without the need for explicit programming.
                        Put differently, machine learning refers to the development of systems that can learn and develop on their own through experience or examples, enabling them to classify data, make predictions, or carry out tasks without explicit guidance.
                    </p>
                    """,
                    unsafe_allow_html=True
            )
            st.markdown("Follow me on:")
            social_media_buttons()
        
        st.title("Evaluation Analysis")
        
        td = st.number_input("Enter the tD value", min_value=0.05, max_value=10000.0, step=0.01)
        
        lr = joblib.load("linear_prediction")
        wed_lr = lr.predict([[float(td)]])
        rf = joblib.load("random_prediction")
        wed_rf = rf.predict([[float(td)]])
        xg = joblib.load("xg_prediction")
        wed_xg = xg.predict([[float(td)]])
        
        new_model = pd.DataFrame({"Dimensionless Time Influx": td, "Dimensionless Water Linear Reg": wed_lr[0], "Dimensionless Water Influx XgBoost": wed_xg[0], "Dimensionless Water Influx Random Forest": wed_rf[0]}, index=[0])
        st.table(new_model)
        
        model = st.selectbox("Select a Model to use", ["Linear Regression", "XGBoost Regression", "Random Forest Regression"])
        if model == "Linear Regression":
            wed = wed_lr    
        elif model == "XGBoost Regression":
            wed = wed_xg
        else:
            wed = wed_rf
            
        B = st.selectbox("Select the type of Water Influx ", ["Constant", "Not Specified"])
        if B == "Constant":
            b = st.number_input("Input the water influx constant (in bbl/psi)", min_value=0.0,
                                            max_value=1000000.0, step=0.01)
            
        else:
            c = st.number_input("Enter the total compressibility value", min_value=0.000000000000,
                                            max_value=1000000.0, step=0.000000000001)
            r = st.number_input("Enter the aquifer raduis", min_value=0.0,
                                            max_value=1000000.0, step=0.01)
            
            flux_0 = st.number_input("Enter the Porosity", min_value=0.0,
                                            max_value=1000000.0, step=0.01, key=1)
            
            
        
            h = st.number_input("Enter the thickness of the aquiver in (feet)", min_value=0.0,
                                            max_value=1000000.0, step=0.01, key=2)
            
            teta = st.selectbox("Select the Enchroachment", ["Full Reservoir(360deg)", "Semi Reservoir(180deg)", "an enchroachment angle(90deg)"])
            
            
            if teta == "Full Reservoir(360deg)":
                data = float(360)

            elif teta == "Semi Reservoir(180deg)":
                data = float(180)
                
            else:
                data = float(90)
            
        
            f = data/float(360)
            
            new_b = 1.119 * flux_0 * float(c) * float(r**2) * f * float(h)
            my_new2 = pd.DataFrame({"Total Compressibility": round(int(c),2), "Aquaver Radius":  round(int(r),2), "Reservior Value": f"{data} degrees", "Thickness": round(int(h),2), "Porosity": round(float(h),2), "Water Inlfux Constant": round(new_b,3)}, index=[0])
            st.table(my_new2)
            
            
        
            
        pr = st.number_input("Enter the Resevoir Pressure(pr) (in psi)", min_value=0.0,
                                        max_value=1000000.0, step=0.01)
        pa = st.number_input("Enter the Aquifer Pressure(pa) (in psi)", min_value=0.0,
                                        max_value=1000000.0, step=0.01)
        p = (pr - pa)
        
         
            
        if B == "Constant" and p != 0.0 and b != 0.0:
            my_new = pd.DataFrame({"Water Influx Constant": round(int(b), 2), "Pressure Drop": round(int(p),2),  "Dimensionless Water inFlux": int(wed)}, index=[0])
            st.table(my_new)
            
            we = round(b,2) * round(p,2) * round(wed,2)
            st.markdown(
                f"""
                    <div style='text-align: center'>
                        <h3> Using Water Influx Formulation (We) </h3>
                        <p> We = B * (change in P) * Wed </p>
                        <p> We = {we}bbl </p>
                    </div>
                """,
                unsafe_allow_html=True)
        elif B == "Not Specified" and c!= 0.0 and r != 0.0 and h != 0.0:
            my_new = pd.DataFrame({"Total Compressibility": round(int(c),2), "Aquaver Radius":  round(int(r),2), "Pressure Drop": round(int(p),2), "Reservior Value": f"{data} degrees", "Dimensionless Water inFlux": round(int(wed),2)}, index=[0])
            st.table(my_new)
            
            we =  int(new_b) * round(p,2) * round(wed,2)
            st.markdown(
                f"""
                    <div style='text-align: center'>
                        <h3> Using Water Influx Formulation (We) </h3>
                        <div style='display: flex; justify-content: space-evenly'>
                            <p> We = B * (change in P) * Wed </p>
                            <span> Where B = 1.119 * c * r^2 * f * h </span>
                        </div>
                        <p> We = {we}bbl </p>
                    </div>
                """,
                unsafe_allow_html=True)


@st.cache_data
def draw_graph(lr_mae, rf_mae, xg_mae, lr_rmse, rf_rmse, xg_rmse):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Define the algorithms and their corresponding mean errors and root mean errors
    algorithms = ['LR', "RF", "XG"]
    mean_errors = [lr_mae, rf_mae, xg_mae]
    root_mean_errors = [lr_rmse, rf_rmse, xg_rmse]
    # avr_mean_errors = [lr_ame, rf_ame, svr_ame, cnn_avr_mean, lstm_ame, xg_ame, gb_ame]

    x = np.arange(len(algorithms))

    bar_width = 0.25

    # Plot the bar chart
    plt.figure(figsize=(6, 4))
    plt.grid(True)
    plt.bar(x, mean_errors, width=0.4, label='Mean Error')
    plt.bar(x + bar_width, root_mean_errors, width=bar_width, label='Root Mean Error')
    # plt.bar(x, root_mean_errors, width=0.4, label='Root Mean Error')

    # Add x-axis labels
    # plt.xticks(x, algorithms)
    # Set the x-axis ticks and labels
    plt.xticks(x + bar_width / 2, algorithms)

    # Set plot labels and title
    plt.xlabel('Algorithms')
    plt.ylabel('Error')
    plt.title('Mean Error and Root Mean Error of Algorithms')

    # Add a legend
    plt.legend()
    st.pyplot()


@st.cache_data
def filter_data(df, search_query):
    # Filter the DataFrame based on the search query
    if search_query:
        filtered_df = df[["tD", "WeD"]].apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(),
                                              axis=1)
        df_filtered = df[["tD", "WeD"]][filtered_df]
    else:
        df_filtered = df[["tD", "WeD"]]
    return df_filtered

def social_media_buttons():
    """
    Renders a custom component with social media icons and links.
    """
    html_code = """
    <style>
    .social-media-container {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
    }
    .social-media-icon {
        width: 32px;
        height: 32px;
        margin: 0 5px;
    }
    </style>
    <div class="social-media-container">
        <a href="https://www.linkedin.com/in/emmanuel-esin-957834200/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/LinkedIn_icon.svg/2048px-LinkedIn_icon.svg.png" height="5px" alt="LinkedIn" class="social-media-icon">
        </a>
        <a href="https://twitter.com/esin_professor" target="_blank">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQbRTyD34UVm3UCNMqYF86xeHdWpLV6Jnvsg&usqp=CAU" alt="Twitter" class="social-media-icon" height="5px">
        </a>
    </div>
    """
    st.components.v1.html(html_code, height=50)
    
hide_st_style = """
    <style>
    nav {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
