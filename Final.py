import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import EDA as EDA
import Kmeans as km

st.set_page_config(
    page_title="Food Nutrition System",
    layout="wide"
)

DATA_PATH = r"venv/Project_3/synthetic_food_dataset1_category_Kmeans.csv"
df = pd.read_csv(DATA_PATH)

@st.cache_resource
def train_xgboost():
    FEATURES = [
    "Calories", "Protein", "Fat", "Carbs", "Sugar", "Fiber",
    "Sodium", "Cholesterol", "Water_Content",
    "Glycemic_Index", "Is_Vegan", "Is_Gluten_Free"
]

    TARGET = "Food_Name"

    X = df[FEATURES]
    y = df[TARGET]

# -------------------- Encode Target --------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("Classes:", le.classes_)
    print("\nClass Distribution:\n", pd.Series(y).value_counts())

    # -------------------- Train-Test Split --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.30,
        random_state=42,
        stratify=y_enc
    )

    # -------------------- Pipeline (Scaler + XGBoost) --------------------
    pipeline = Pipeline([
        ("xgb", XGBClassifier(
            max_depth=2,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            num_class=len(le.classes_),
            eval_metric="mlogloss",
            random_state=42
        ))
    ])

    # -------------------- Train Model --------------------
    pipeline .fit(X_train, y_train)
    return pipeline, le, FEATURES

xgb_model, cat_encoder, FEATURE_COLS = train_xgboost()

st.sidebar.title("Food Nutrition")
page = st.sidebar.radio(
    "Navigate",
    [
        "Smart Dietary Applications",
        "Health Monitoring Tools",
        "Education Platform"
    ]
)

df["Nutrition_Label_Text"] = pd.Categorical(
    df["Nutrition_Label_Text"],
    categories=["Poor", "Moderate", "High", "Very High"],
    ordered=True
)

if page == "Education Platform":
    st.header("Food Nutrition")

    st.subheader("Food Classification based on High sugar, Good fiber, High protein, Low GI")
    food = st.selectbox("Select Food", df["Food_Name"].unique())
    row = df[df["Food_Name"] == food].iloc[0]

    reasons = []

    if row["Sugar"] > 15:
        reasons.append("High sugar")
    if row["Fiber"] >= 5:
        reasons.append("Good fiber")
    if row["Protein"] >= 10:
        reasons.append("High protein")
    if row["Glycemic_Index"] <= 55:
        reasons.append("Low GI")

    reason_text = ", ".join(reasons) if reasons else "No significant nutritional flags"

    st.write(f"{food} has {reason_text}.")

    st.subheader("Food classification based on the nutrition level Poor, Moderate, High and Very high")
    level_map = {
        "Poor": 0,
        "Moderate": 1,
        "High": 2,
        "Very High": 3
    }

    level_text = st.selectbox(
        "Nutrition Level",
        list(level_map.keys())
    )

    level_value = level_map[level_text]

    result = (
        df[df["Nutrition_Label"] == level_value]     # ✅ boolean condition
        .sort_values("Nutrition_Score_Value", ascending=False)  # ✅ column name
        .drop_duplicates(subset="Food_Name")          # ✅ column name
    )

    display_df = result[
    ["Food_Name", "Nutrition_Label_Text", "Nutrition_Score_Normalized"]
    ].rename(columns={
    "Food_Name": "Food Name",
    "Nutrition_Label_Text": "Nutrition Level",
    "Nutrition_Score_Normalized": "Nutrition Score Percentile"
    })

    st.dataframe(display_df, hide_index= True)

    st.subheader("Nutrition per 100 gram of food")
    mean_df = (
        df.groupby("Food_Name", as_index=False)
          .agg(
              Mean_Calories=("Calories", "mean"),
              Mean_Protein=("Protein", "mean"),
              Mean_Fiber=("Fiber", "mean")
          ).rename(columns={
            "Food_Name": "Food Name",
            "Mean_Calories": "Calories",
            "Mean_Protein": "Protein",
            "Mean_Fiber": "Fiber" 
          })
    )

    st.dataframe(mean_df, hide_index=True)

if page == "Smart Dietary Applications":

    df["Meal_Type"] = df["Meal_Type"].fillna("snack")

    st.subheader("Daily Calorie Distribution")

    total_cal = st.slider(
        "Total Daily Calories",
        min_value=800,
        max_value=2500,
        value=1500,
        step=100
    )

    breakfast_cal = int(total_cal * 0.25)
    lunch_cal = int(total_cal * 0.35)
    dinner_cal = int(total_cal * 0.30)
    snack_cal = total_cal - (breakfast_cal + lunch_cal + dinner_cal)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Breakfast", breakfast_cal)
    col2.metric("Lunch", lunch_cal)
    col3.metric("Dinner", dinner_cal)
    col4.metric("Snack", snack_cal)

    EXCLUDED_FOODS_MAIN_MEALS = [
        "ice cream", "donut", "doughnut", "cake", "pastry",
        "brownie", "cookie", "chocolate", "sweet",
        "muffin", "cupcake", "pudding", "gelato"
    ]

    DEFAULT_PORTION_GRAMS = {
    "Breakfast": 250,
    "Lunch": 350,
    "Dinner": 300,
    "Snack": 150
    }

    def meal_calories(cal_per_100g, portion_g):
        return (cal_per_100g * portion_g) / 100

    def is_allowed_for_meal(food_name, meal_type):
        name = food_name.lower()
        if meal_type in ["Breakfast", "Lunch", "Dinner"]:
            return not any(junk in name for junk in EXCLUDED_FOODS_MAIN_MEALS)
        return True

    def filter_by_calories(df, target_cal, meal_type, tolerance=0.15):
        portion = DEFAULT_PORTION_GRAMS.get(meal_type, 250)

        df = df.copy()

        # Calculate actual meal calories
        df["Meal_Calories"] = (
            df["Calories"] * portion / 100
        )

        lower = target_cal * (1 - tolerance)
        upper = target_cal * (1 + tolerance)

        filtered = df[
            (df["Meal_Calories"] >= lower) &
            (df["Meal_Calories"] <= upper)
        ]

        # Exclude junk foods for main meals
        filtered = filtered[
            filtered["Food_Name"].apply(
                lambda x: is_allowed_for_meal(x, meal_type)
            )
        ]

        filtered = filtered.drop_duplicates(subset=["Food_Name"])    

        # Safe sorting
        sort_cols = []
        if "Nutrition_Score_Normalized" in filtered.columns:
            filtered = filtered.sort_values(
            by="Nutrition_Score_Normalized",
            ascending=False
        )

        return filtered   

    def generate_meal_plan(df):
        return {
            "Breakfast": filter_by_calories(
                df[df["Meal_Type"] == "breakfast"],
                breakfast_cal,
                "Breakfast"
            ),

            "Lunch": filter_by_calories(
                df[df["Meal_Type"] == "lunch"],
                lunch_cal,
                "Lunch"
            ).head(5),

            "Dinner": filter_by_calories(
                df[df["Meal_Type"] == "dinner"],
                dinner_cal,
                "Dinner"
            ),

            "Snack": filter_by_calories(
                df[df["Meal_Type"] == "snack"],
                snack_cal,
                "Snack"
            ),
        }

    st.subheader("Generated Daily Meal Plan option with nutritional data per 100 gram")

    meal_plan = generate_meal_plan(df)

    DISPLAY_COLUMNS = [
    "Calories",
    "Protein",
    "Sugar",
    "Fiber",
    "Fat",
    "Nutrition_Score_Normalized"
]

    def aggregate_by_food_name(df):
        agg = (
        df.groupby("Food_Name", as_index=False)[DISPLAY_COLUMNS]
          .mean()
          .round(2)
    )
        if "Nutrition_Score_Normalized" in agg.columns:
            agg = agg.sort_values(
            by="Nutrition_Score_Normalized",
            ascending=False
        )

        return agg

    for meal, items in meal_plan.items():
        st.markdown(f"### {meal}")
        st.write("calculate the portion as per the calories mentioned")

        if not items.empty:
            items = items.copy()
            display_df = aggregate_by_food_name(items)        

            portion = DEFAULT_PORTION_GRAMS[meal]
            items["Portion (g)"] = portion
            items["Meal Calories"] = (
                items["Calories"] * portion / 100
            )

            st.dataframe(
            display_df,
            hide_index=True
            )
        else:
            st.warning("No suitable food found")

if page == "Health Monitoring Tools":
    
    st.header("Food Classification (Manual Input)")

    with st.form("food_form"):
        user_input = {
            "Calories": st.number_input("Calories"),
            "Protein": st.number_input("Protein"),
            "Fat": st.number_input("Fat"),
            "Carbs": st.number_input("Carbs"),
            "Sugar": st.number_input("Sugar"),
            "Fiber": st.number_input("Fiber"),
            "Sodium": st.number_input("Sodium"),
            "Cholesterol": st.number_input("Cholesterol"),
            "Glycemic_Index": st.number_input("Glycemic Index"),
            "Water_Content": st.number_input("Water Content"),
            "Is_Vegan": int(st.checkbox("Is Vegan")),
            "Is_Gluten_Free": int(st.checkbox("Is Gluten Free"))
        }

        classify_manual = st.form_submit_button("Classify")

    if classify_manual:
        input_df = pd.DataFrame([user_input])[FEATURE_COLS]
        pred = xgb_model.predict(input_df)
        category = cat_encoder.inverse_transform(pred)[0]

        st.success(f"Predicted Food Category: {category}")

    st.header("Food Classification (CSV Upload)")

    with st.form("csv_upload_form"):
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"]
        )
    
        classify_csv = st.form_submit_button("Upload & Classify")
    
    if classify_csv:
        if uploaded_file is None:
            st.error("Please upload a CSV file")
            st.stop()
    
        df = pd.read_csv(uploaded_file)
    
        required_columns = [
            "Calories",
            "Protein",
            "Fat",
            "Carbs",
            "Sugar",
            "Fiber",
            "Sodium",
            "Cholesterol",
            "Glycemic_Index",
            "Water_Content",
            "Is_Vegan",
            "Is_Gluten_Free"
        ]
         
        missing_cols = set(required_columns) - set(df.columns)
    
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()
    
        st.success("CSV loaded successfully")
    
        # Ensure correct column order
        input_df = df[FEATURE_COLS]
    
        preds = xgb_model.predict(input_df)
        category1 = cat_encoder.inverse_transform(preds)
        st.success("Food classification completed")        

        st.success(f"Predicted Food:")
        for i, cat in enumerate(category1, start=1):
            st.write(f"Food {i}: {cat}")

    
    
    
    
    