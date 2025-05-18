import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Load and clean data function
def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Remove "Ask For Price"
    df = df[df['Price'] != 'Ask For Price']

    # Clean Price
    df['Price'] = df['Price'].str.replace('₹', '').str.replace('Lakh', '00000').str.replace(',', '').str.strip()
    df = df[df['Price'].str.replace('.', '').str.isnumeric()]
    df['Price'] = df['Price'].astype(float)

    # Clean kms_driven
    df = df[df['kms_driven'].notna()]
    df['kms_driven'] = df['kms_driven'].str.replace(' kms', '').str.replace(',', '').str.strip()
    df = df[df['kms_driven'].str.isnumeric()]
    df['kms_driven'] = df['kms_driven'].astype(int)

    # Clean year
    df['year'] = df['year'].astype(str).str.strip()
    df = df[df['year'].str.isnumeric()]
    df['year'] = df['year'].astype(int)
    df['age'] = 2025 - df['year']

    return df

# Prepare features for modeling
def prepare_features(df):
    df = df[['company', 'fuel_type', 'kms_driven', 'age', 'Price']]
    df = pd.get_dummies(df, drop_first=True)
    return df

# Train model
def train_model(df_features):
    X = df_features.drop('Price', axis=1)
    y = df_features['Price']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model, X.columns

# Prepare single input for prediction
def prepare_single_input(company, fuel_type, kms_driven, year, feature_columns):
    age = 2025 - year
    input_dict = {
        'kms_driven': [kms_driven],
        'age': [age]
    }

    # Add one-hot encoding columns
    for col in feature_columns:
        if col.startswith('company_'):
            input_dict[col] = [1 if col == f'company_{company}' else 0]
        elif col.startswith('fuel_type_'):
            input_dict[col] = [1 if col == f'fuel_type_{fuel_type}' else 0]
        elif col not in ['kms_driven', 'age']:
            input_dict[col] = [0]

    input_df = pd.DataFrame(input_dict)
    return input_df

def main():
    data_path = r"C:\Users\deepa\OneDrive\Desktop\TODAY PROJECT\quikr_car.csv"
    df = load_and_clean_data(data_path)
    df_features = prepare_features(df)
    model, feature_columns = train_model(df_features)

    # Batch prediction for whole CSV
    df_original = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']].copy()
    X_all = df_features.drop('Price', axis=1)
    df_original['Predicted_Price'] = model.predict(X_all).astype(int)
    
    # Save to Documents folder to avoid permission issues
    output_csv = r"C:\Users\deepa\Documents\car_price_predictions.csv"
    df_original.to_csv(output_csv, index=False)
    print(f"\n✅ Batch prediction saved to: {output_csv}")

    # Single car prediction from terminal input
    print("\nEnter details for single car price prediction:")

    user_company = input("Company (e.g. Hyundai): ").strip()
    user_fuel = input("Fuel Type (Petrol/Diesel): ").strip().capitalize()
    user_kms = int(input("Kilometers Driven: "))
    user_year = int(input("Year of Purchase: "))

    user_input_df = prepare_single_input(user_company, user_fuel, user_kms, user_year, feature_columns)
    predicted_price = model.predict(user_input_df)[0]

    print(f"\n💰 Predicted Price for your car: ₹{int(predicted_price):,}")

if __name__ == "__main__":
    main()
