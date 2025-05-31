# 📦 Import required libraries:
# These libraries are essential for building the app interface, handling data,
# loading models, creating plots, and calculating similarity between product features.
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 🎨 Set background function:
# This function applies a custom background image to the Streamlit app using raw HTML and CSS.
# It enhances UI aesthetics by embedding an external image URL.
def set_bg(image_url):
    st.markdown(f"""
    <style>
    .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
            }}        
    </style>
    """,
    unsafe_allow_html=True)

# -------------------------------
# Data Preprocessing
# -------------------------------

# ✅ Define the preprocessing pipeline function
def preprocess_pipeline(df):
    # Avoid recalculating if already present
    if 'total_clicks' not in df.columns:
        total_clicks = df.groupby('session_id')['order'].count().reset_index()
        total_clicks.rename(columns={'order': 'total_clicks'}, inplace=True)
        df = df.merge(total_clicks, on='session_id', how='left')

    if 'browsing_depth' not in df.columns:
        browsing_depth = df.groupby('session_id')['page'].max().reset_index()
        browsing_depth.rename(columns={'page': 'browsing_depth'}, inplace=True)
        df = df.merge(browsing_depth, on='session_id', how='left')

    # Create weekday and weekend flags
    if {'year', 'month', 'day'}.issubset(df.columns):
        df['weekday'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.dayofweek
        df['weekend'] = (df['weekday'] >= 5).astype(int)
        df.drop(columns=[col for col in ['year'] if col in df.columns], inplace=True)

    # Drop unnecessary columns safely
    df.drop(columns=[col for col in ['session_id', 'page2_clothing_model'] if col in df.columns], inplace=True)

    return df


# ✅ Save the pipeline function as a pickle
with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/preprocessing_pipeline.pkl", "wb") as f:
    pickle.dump(preprocess_pipeline, f)

print("✅ Preprocessing pipeline pickled successfully!")

# -------------------------------
# Mappings for encoding
# -------------------------------
country_mapping = {
    "Australia": 1, "Austria": 2, "Belgium": 3, "British Virgin Islands": 4, "Cayman Islands": 5,
    "Christmas Island": 6, "Croatia": 7, "Cyprus": 8, "Czech Republic": 9, "Denmark": 10,
    "Estonia": 11, "unidentified": 12, "Faroe Islands": 13, "Finland": 14, "France": 15,
    "Germany": 16, "Greece": 17, "Hungary": 18, "Iceland": 19, "India": 20,
    "Ireland": 21, "Italy": 22, "Latvia": 23, "Lithuania": 24, "Luxembourg": 25,
    "Mexico": 26, "Netherlands": 27, "Norway": 28, "Poland": 29, "Portugal": 30,
    "Romania": 31, "Russia": 32, "San Marino": 33, "Slovakia": 34, "Slovenia": 35,
    "Spain": 36, "Sweden": 37, "Switzerland": 38, "Ukraine": 39, "United Arab Emirates": 40,
    "United Kingdom": 41, "USA": 42, "biz (.biz)": 43, "com (.com)": 44,
    "int (.int)": 45, "net (.net)": 46, "org (*.org)": 47
}

main_category_mapping = {
    "trousers": 1, "skirts": 2, "blouses": 3, "sale": 4
}

colour_mapping = {
    "beige": 1, "black": 2, "blue": 3, "brown": 4, "burgundy": 5,
    "gray": 6, "green": 7, "navy blue": 8, "of many colors": 9,
    "olive": 10, "pink": 11, "red": 12, "violet": 13, "white": 14
}

location_mapping = {
    "top left": 1, "top in the middle": 2, "top right": 3,
    "bottom left": 4, "bottom in the middle": 5, "bottom right": 6
}

model_photo_mapping = {
    "en face": 1, "profile": 2
}


# -------------------------------
# Streamlit Input Function
# -------------------------------
def get_customer_session_details_classification():
    st.header("📊 Input the Customer Session to Predict the Magic!")

    with st.form("customer_session_form"):
        year = st.number_input("Year",min_value=2008)
        month = st.selectbox("Month", list(range(1, 13)))
        day = st.number_input("Day", min_value=1, max_value=31)
        order = st.number_input("Order", min_value=1)
        country = st.selectbox("Country", list(country_mapping.keys()))
        session_id = st.number_input("Session ID",min_value=1000)
        page1_main_category = st.selectbox("Main Category", list(main_category_mapping.keys()))
        colour = st.selectbox("Colour", list(colour_mapping.keys()))
        location = st.selectbox("Location", list(location_mapping.keys()))
        model_photography = st.selectbox("Model Photo", list(model_photo_mapping.keys()))

        price = st.number_input("Price", min_value=0.0)
        page = st.number_input("Page", min_value=1)
        total_clicks = st.number_input("Total Clicks", min_value=1)

        submitted = st.form_submit_button("Submit")

        if submitted:
            # Encode selections
            data = {
                "year":[year],
                "month": [month],
                "day": [day],
                "order": [order],
                "country": [country_mapping[country]],
                "session_id":[session_id],
                "page1_main_category": [main_category_mapping[page1_main_category]],
                "colour": [colour_mapping[colour]],
                "location": [location_mapping[location]],
                "model_photography": [model_photo_mapping[model_photography]],
                "price": [price],
                "page": [page],
                "total_clicks": [total_clicks],
            }

            uploaded_file = pd.DataFrame(data)
            st.success("Customer session data captured successfully!")
            return uploaded_file

    return None

# ✅ Function to collect form inputs
def get_customer_session_details_regression():
    st.header("📊 Input the Customer Session to Predict the Magic!")

    with st.form("customer_session_form"):
        year = st.number_input("Year", min_value=2008)
        month = st.selectbox("Month", list(range(1, 13)))
        day = st.number_input("Day", min_value=1, max_value=31)
        order = st.number_input("Order", min_value=1)
        country = st.selectbox("Country", list(country_mapping.keys()))
        session_id = st.number_input("Session ID", min_value=1000)
        page1_main_category = st.selectbox("Main Category", list(main_category_mapping.keys()))
        colour = st.selectbox("Colour", list(colour_mapping.keys()))
        location = st.selectbox("Location", list(location_mapping.keys()))
        model_photography = st.selectbox("Model Photo", list(model_photo_mapping.keys()))
        price_2 = st.number_input("Price_2", min_value=0.0)
        page = st.number_input("Page", min_value=1)
        total_clicks = st.number_input("Total Clicks", min_value=1)

        submitted = st.form_submit_button("Submit")

        if submitted:
            user_inputs = [
                year,
                month,
                day,
                order,
                country_mapping[country],
                session_id,
                main_category_mapping[page1_main_category],
                colour_mapping[colour],
                location_mapping[location],
                model_photo_mapping[model_photography],
                price_2,
                page,
                total_clicks
            ]
            return user_inputs

    return None


# Use an image URL or local file path
# For local files, use base64 encoding (see below)
image_url = "https://i.postimg.cc/sD339hYg/colorful-flyer-with-colorful-border-design-confetti-shopping-bags-with-stars-1174726-8729.avif"  # Example URL
set_bg(image_url)

st.image("https://i.postimg.cc/L4NbwgdX/Gemini-Generated-Image-j6ywsgj6ywsgj6yw.png")
st.subheader("📂 Ready to explore? Select your path!")
choice = st.radio("", ["About Click2Customer","Classification","Regression","Clustering"],horizontal=True)

if choice == "About Click2Customer":
        proceed = st.button("Know us")
        if proceed:
            # About Section
            st.title("🛍️ About Us – Click2Customer")
            st.markdown("---")

            st.markdown("""
            **Welcome to Click2Customer – where data meets decision.**

            At Click2Customer, we are a pioneering force in the e-commerce landscape, on par with industry giants like **Amazon**, **Walmart**, and **eBay**.
            Our mission is to harness the power of data to understand and enhance the online shopping experience.
            As a leading-edge platform, Click2Customer transforms raw clickstream data into actionable intelligence—empowering businesses to convert visitors into loyal customers, increase revenue, and deliver hyper-personalized shopping experiences.
            """)

            st.header("📊 Our Vision")
            st.markdown("""
            To revolutionize the way e-commerce businesses understand customer behavior by providing an intelligent, intuitive, and data-driven web application that drives real results.
            """)

            st.header("🚀 What We Do")
            st.markdown("Built using **Streamlit**, our smart and user-friendly web application is designed to elevate your online shopping analytics. Here's how:")

            st.subheader("✅ 1. Classification: Purchase Prediction")
            st.markdown("""
            We predict whether a visitor will complete a purchase based on their browsing behavior. This allows you to:
            - Understand where customers drop off  
            - Intervene in real-time with tailored promotions  
            - Reduce cart abandonment rates
            """)

            st.subheader("💰 2. Regression: Revenue Forecasting")
            st.markdown("""
            Our models estimate the **potential revenue** each visitor might generate, helping you:
            - Forecast overall business revenue  
            - Optimize marketing spend  
            - Prioritize high-value customer segments
            """)

            st.subheader("🧠 3. Clustering: Customer Segmentation")
            st.markdown("""
            Using advanced clustering techniques, we group customers based on online behavior. This enables:
            - Precision-targeted marketing campaigns  
            - Personalized product recommendations  
            - Smarter inventory and pricing strategies
            """)

            st.header("💡 Why Click2Customer?")
            st.markdown("""
            - **AI-Powered Insights**: Leverage machine learning models for predictive analytics.  
            - **Real-Time Engagement**: React instantly to customer behaviors with data-driven interventions.  
            - **Business Growth Focus**: Drive more conversions, increase revenue, and maximize customer satisfaction.
            """)

            st.header("🤝 Empowering E-Commerce Excellence")
            st.markdown("""
            Whether you're a startup or a global e-commerce leader, **Click2Customer** gives you the tools to stay ahead of the curve. 
            We transform clickstream chaos into clarity—so every click counts.

            **Join us in shaping the future of online shopping, one click at a time.**
            """)
elif choice == "Classification":
      # Load your preprocessing function from pickle
    with open("Pickled Data/preprocessing_pipeline.pkl", "rb") as f:
        preprocess_pipeline = pickle.load(f)

    # Set the background
    set_bg("https://i.postimg.cc/pLGgYDn7/Pngtree-illustration-of-online-shopping-concept-1389336.png")

    # UI
    st.subheader("📌 Select your data style: Bulk load or single shot?")
    process = st.selectbox("", ['📦 Bulk data – it\'s raining rows and columns!', '🔍 Single query – just a quick peek at the facts'])

    # If bulk data selected
    if process == "📦 Bulk data – it\'s raining rows and columns!":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(df)
                if not df.empty:
                    processed_df = preprocess_pipeline(df)
                    # Optional: Save to file if needed
                    processed_df.to_csv("/Users/somesh-19583/Desktop/Customer conversion/Source data/uploaded_file_bulk.csv", index=False)

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

        uploaded_file = pd.read_csv("/Users/somesh-19583/Desktop/Customer conversion/Source data/uploaded_file_bulk.csv")
        # ---------- Load and preprocess test data ----------
        test_df = pd.read_csv("/Users/somesh-19583/Desktop/Customer conversion/Source data/uploaded_file_bulk.csv")


        # ---------- Prepare features and target ----------
        # Make sure your test_df includes the target column 'price_2'
        X_test = test_df.drop('price_2', axis=1)
        y_test = test_df['price_2']

        # ---------- Load the pickled Gradient Boosting regression model ----------
        with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/decision_tree_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # ---------- Predict on the test data ----------
        y_pred = pd.DataFrame(loaded_model.predict(X_test))
        Submit = st.button("Proceed")
        if Submit:
            st.write("📥 Predictions for your uploaded data are now available!")
            st.write(y_pred)

    else:
        # ✅ Main app
        st.title("🧮 Predict Purchase")

        # Load the Gradient Boosting model
        with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/decision_tree_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # Get user input
        inp = get_customer_session_details_classification()

        # ✅ Check that inp is not None and not empty
        if inp is not None and not inp.empty:
            
            # Run preprocessing
            processed_df = preprocess_pipeline(inp)

            # Prediction
            result = loaded_model.predict(processed_df)

            if result[0] == 1:
                st.success("📈 Prediction: This session will end in a successful purchase.")
            else:
                st.warning("🧐 Prediction: The customer won’t complete the purchase.")



elif choice=="Clustering":
        set_bg("https://i.postimg.cc/gkhm7r6v/freepik-animated-shopping-icons-cascade-down-the-right-bor-29726.jpg")

        # ------------------------------------------
        # Load pre-trained models
        # ------------------------------------------
        with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/pca.pkl", "rb") as f:
            pca = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/kmeans.pkl", "rb") as f:
            kmeans = pickle.load(f)

        # ------------------------------------------
        # Define mappings
        # ------------------------------------------
        main_category_mapping = {
            1: "trousers", 2: "skirts", 3: "blouses", 4: "sale"
        }

        colour_mapping = {
            1: "beige", 2: "black", 3: "blue", 4: "brown", 5: "burgundy", 6: "gray",
            7: "green", 8: "navy blue", 9: "of many colors", 10: "olive",
            11: "pink", 12: "red", 13: "violet", 14: "white"
        }

        numerical_features = ['page1_main_category', 'colour', 'price']

        # ------------------------------------------
        # Recommendation function
        # ------------------------------------------
        def recommend_products(product_name, df, num_recommendations=5):
            if product_name not in df["page2_clothing_model"].values:
                return None

            product_cluster = df[df["page2_clothing_model"] == product_name]["Cluster"].values[0]
            same_cluster_df = df[df["Cluster"] == product_cluster].reset_index(drop=True)

            try:
                product_index = same_cluster_df[same_cluster_df["page2_clothing_model"] == product_name].index[0]
            except IndexError:
                return None

            cluster_features = same_cluster_df[numerical_features]
            similarity = cosine_similarity(cluster_features, cluster_features)

            similar_indices = np.argsort(similarity[product_index])[::-1]
            similar_indices = [i for i in similar_indices if i != product_index][:num_recommendations]

            recommendations = same_cluster_df.iloc[similar_indices].copy()
            recommendations = recommendations.drop_duplicates(subset=["page2_clothing_model"])

            recommendations["page1_main_category"] = recommendations["page1_main_category"].map(main_category_mapping)
            recommendations["colour"] = recommendations["colour"].map(colour_mapping)

            return recommendations[["page2_clothing_model", "colour", "price", "page1_main_category"]]

        # ------------------------------------------
        # Streamlit App
        # ------------------------------------------
        st.header("🧩 Smart Clustering & Personalized Product Insights")
        st.write("🤖 Power Up Your Product Insights – Just Upload to Get Started!")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        input_product = st.text_input("Enter Product Name (e.g., A12)").upper()

        # Capture button press properly
        run_button = st.button("🎯 Reveal Clusters & Top Picks")

        if uploaded_file and input_product and run_button:
            try:
                df = pd.read_csv(uploaded_file)
                df = df.dropna(subset=numerical_features + ["page2_clothing_model"])

                # Apply models
                scaled_data = scaler.transform(df[numerical_features])
                df["Cluster"] = kmeans.predict(scaled_data)

                # PCA transformation
                pca_features = pca.transform(scaled_data)
                df["PCA1"] = pca_features[:, 0]
                df["PCA2"] = pca_features[:, 1]

                # Plotting
                st.header("📊 PCA Cluster Plot")
                fig, ax = plt.subplots(figsize=(10,5))
                scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="tab10", s=50)
                ax.set_title("KMeans Clustering (PCA Projection)")
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                ax.grid("True")

                # Render in Streamlit
                st.pyplot(fig)

                # Map categorical values for plotting
                df["colour_mapped"] = df["colour"].map(colour_mapping)
                df["main_category_mapped"] = df["page1_main_category"].map(main_category_mapping)

                # Prepare data for charts
                category_counts = df["main_category_mapped"].value_counts()
                colour_counts = df["colour_mapped"].value_counts()

                # Create large figure with 3 subplots
                fig, axs = plt.subplots(1, 3, figsize=(30, 12))  # Wider and taller for clarity
                fig.suptitle("Product Data Visualizations", fontsize=32)  # Bigger title

                # Pie chart: product category distribution
                axs[0].pie(
                    category_counts,
                    labels=category_counts.index,
                    autopct="%1.1f%%",
                    startangle=140,
                    textprops={"fontsize": 24}  # Set font size for pie labels
                )
                axs[0].set_title("Product Category Distribution (Pie Chart)", fontsize=24)

                # Bar chart: color counts
                axs[1].bar(colour_counts.index, colour_counts.values, color="skyblue")
                axs[1].set_title("Color Distribution (Bar Chart)", fontsize=24)
                axs[1].set_xlabel("Colour", fontsize=18)
                axs[1].set_ylabel("Count", fontsize=18)
                axs[1].tick_params(axis='x', labelsize=14, rotation=45)
                axs[1].tick_params(axis='y', labelsize=14)

                # Histogram: price distribution
                axs[2].hist(df["price"], bins=20, color="orange", edgecolor="black")
                axs[2].set_title("Price Distribution (Histogram)", fontsize=24)
                axs[2].set_xlabel("Price", fontsize=18)
                axs[2].set_ylabel("Frequency", fontsize=18)
                axs[2].tick_params(axis='x', labelsize=14)
                axs[2].tick_params(axis='y', labelsize=14)
                axs[2].grid(True)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Show the plot in Streamlit
                st.pyplot(fig)

                # Recommendation
                st.subheader("🔍 Recommended Products")
                recommendations = recommend_products(input_product, df, num_recommendations=5)

                if recommendations is not None and not recommendations.empty:
                    st.dataframe(recommendations.reset_index(drop=True))
                    st.success("🚀 Success! We've unlocked your clusters and curated personalized recommendations !!!")
                else:
                    st.warning("❌ Oops! We couldn't find any matches for this product.")

            except Exception as e:
                st.error(f"Error: {e}")
else:

    # Load your preprocessing function from pickle
    with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/preprocessing_pipeline.pkl", "rb") as f:
        preprocess_pipeline = pickle.load(f)

    # Set the background
    set_bg("https://i.postimg.cc/hP1R7gQP/Pngtree-3d-render-online-shopping-with-1438339.jpg")

    # UI
    st.subheader("📌 What kind of data are you rocking right now?")
    process = st.selectbox("", ['📦 Bulk data – it\'s raining rows and columns!', '🔍 Single query – just a quick peek at the facts'])

    # If bulk data selected
    if process == "📦 Bulk data – it\'s raining rows and columns!":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(df)
                if not df.empty:
                    processed_df = preprocess_pipeline(df)
                    # Optional: Save to file if needed
                    processed_df.to_csv("/Users/somesh-19583/Desktop/Customer conversion/Source data/uploaded_file_bulk.csv", index=False)

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
        uploaded_file = pd.read_csv("/Users/somesh-19583/Desktop/Customer conversion/Source data/uploaded_file_bulk.csv")
        # ---------- Load and preprocess test data ----------
        test_df = pd.read_csv("/Users/somesh-19583/Desktop/Customer conversion/Source data/uploaded_file_bulk.csv")


        # ---------- Prepare features and target ----------
        # Make sure your test_df includes the target column 'price_2'
        X_test = test_df.drop('price', axis=1)
        y_test = test_df['price']

        # ---------- Load the pickled Gradient Boosting regression model ----------
        with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/gradient_boostl.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # ---------- Predict on the test data ----------
        y_pred = pd.DataFrame(loaded_model.predict(X_test))
        Submit = st.button("Proceed")
        if Submit:
            st.write("🧾 Below are the purchase amounts for the customers from your uploaded file.")
            st.write(y_pred)

    else:
        st.title("🧮 Estimate Your Purchase Cost")

        # Load the Gradient Boosting model
        with open("/Users/somesh-19583/Desktop/Customer conversion/Pickled Data/gradient_boostl.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # Get user input (returns a list)
        inp = get_customer_session_details_regression()

        # Check if user submitted input
        if inp:
            # Convert to DataFrame
            input_df = pd.DataFrame([inp], columns=[
                "year", "month", "day", "order", "country", "session_id",
                "page1_main_category", "colour", "location", "model_photography",
                "price_2", "page", "total_clicks"
            ])

            # Preprocess the input
            processed_df = preprocess_pipeline(input_df)

            # Make prediction
            result = loaded_model.predict(processed_df)

            # Display result
            if result[0] != 0:
                st.success(f"💰 Your Estimated Purchase: **${result[0]:.2f}**")
            else:
                st.info("The model predicts a value of 0. No estimated amount.")
