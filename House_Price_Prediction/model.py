import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def login():
    st.markdown("## 🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("❌ Invalid credentials!")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()


theme = st.radio("🌗 Select Theme", ["Light", "Dark"], horizontal=True)

# Theme Styling
def set_background_and_style():
    if theme == "Light":
        bg_image = "https://images.boldsky.com/img/2024/06/paintthewall-1718737981.jpg"
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('{bg_image}');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: #000000;
            }}
            .main-box {{
                background-color: rgba(255, 255, 255, 0.85);
                padding: 2rem;
                border-radius: 15px;
                max-width: 900px;
                margin: 2rem auto;
                color: #1f77b4;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            label, .stSelectbox label, .stNumberInput label {{
                font-weight: bold !important;
                color: #1f77b4 !important;
            }}
            button[kind="primary"] {{
                background-color: #1f77b4;
                color: white;
                font-weight: bold;
                border-radius: 8px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .main-box {
                background-color: #2C2C2C;
                padding: 2rem;
                border-radius: 15px;
                max-width: 900px;
                margin: 2rem auto;
                color: #FFFFFF;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            label, .stSelectbox label, .stNumberInput label {
                font-weight: bold !important;
                color: #1f77b4 !important;
            }
            button[kind="primary"] {
                background-color: #1f77b4;
                color: white;
                font-weight: bold;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

set_background_and_style()

# ------------------- Load & Preprocess Dataset -------------------
df = pd.read_csv('Housing.csv')

label_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                 'airconditioning', 'prefarea', 'furnishingstatus']
label_encoders = {}

for col in label_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    joblib.dump(le, f'label_encoder_{col}.pkl')

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------- Streamlit Interface -------------------
with st.container():
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.title("🏠 House Price Prediction")

    with st.form("house_form"):
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Area (in sqft)", min_value=1)
            bedrooms = st.number_input("Number of Bedrooms", min_value=1)
            bathrooms = st.number_input("Number of Bathrooms", min_value=1)
            stories = st.number_input("Number of Stories", min_value=1)
            parking = st.number_input("Number of Parking Spaces", min_value=0)

        with col2:
            mainroad = st.selectbox("Main Road Access", ['yes', 'no'])
            guestroom = st.selectbox("Guestroom Available", ['yes', 'no'])
            basement = st.selectbox("Basement Available", ['yes', 'no'])
            hotwaterheating = st.selectbox("Hotwater Heating", ['yes', 'no'])
            airconditioning = st.selectbox("Air Conditioning", ['yes', 'no'])
            prefarea = st.selectbox("Preferred Area", ['yes', 'no'])
            furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

        submit = st.form_submit_button("Predict")

    if submit:
        def encode(col, val):
            le = joblib.load(f'label_encoder_{col}.pkl')
            try:
                return le.transform([val])[0]
            except ValueError:
                return -1

        user_input = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': encode('mainroad', mainroad),
            'guestroom': encode('guestroom', guestroom),
            'basement': encode('basement', basement),
            'hotwaterheating': encode('hotwaterheating', hotwaterheating),
            'airconditioning': encode('airconditioning', airconditioning),
            'parking': parking,
            'prefarea': encode('prefarea', prefarea),
            'furnishingstatus': encode('furnishingstatus', furnishingstatus)
        }])

        user_input = user_input[X.columns]
        predicted_price = model.predict(user_input)[0]
        predicted_price = max(predicted_price, 0)

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size: 24px; font-weight: bold; color: #1f77b4; padding: 10px 0;">
        💰 Predicted House Price: ${predicted_price:,.2f}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.write("📊 **Model Performance:**")
        st.markdown(f"- Mean Squared Error: `{mse:.2f}`")
        st.markdown(f"- R² Score: `{r2:.2f}`")

        # Error Distribution (Residuals Histogram)
        st.subheader("📉 Error Distribution (Residuals Histogram)")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        ax.hist(residuals, bins=20, alpha=0.7, color='blue')
        ax.set_title("Residuals (Actual - Predicted) Distribution")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Download Prediction
        csv_download = user_input.copy()
        csv_download['Predicted_Price'] = predicted_price
        csv = csv_download.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction", data=csv, file_name="prediction.csv", mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)