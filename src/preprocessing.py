import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



# import scaler, encoder, imputer yang telah dibuat di notebook

def format_cleaning(X):
  X = X.copy()

  #Handling Inconsistencies in 'fuel_type'
  X['fuel_type'] = X['fuel_type'].replace({'–': 'Unknown','not supported': 'Other', np.nan: 'Unknown'})

  #Handling Inconsistencies in 'Transmission'
  trans_series = X['transmission'].astype(str).str.lower().str.strip()
  X['transmission'] = "Unknown"
  X.loc[trans_series.str.contains(r'(?:a\/t|at|automatic|auto[-\s]?shift|cvt|dct|\d+\s*[-]?\s*speed\s*(?:a\/t|at|automatic))', regex=True, na=False), 'transmission'] = "a/t"
  X.loc[trans_series.str.contains(r'(?:m\/t|mt|manual|\d+\s*[-]?\s*speed\s*(?:m\/t|mt|manual))', regex=True, na=False), 'transmission'] = "m/t"
  return X

def extract_fuel(val):
    s = str(val).lower()

    if "hydrogen"in s:
        return "Hydrogen"
    elif "electric" in s:
        return "Electric"
    elif "plug-in hybrid" in s:
        return "Plug-In Hybrid"
    elif "hybrid" in s:
        return "Hybrid"
    elif "diesel" in s:
        return "Diesel"
    elif "gasoline" in s:
        return "Gasoline"
    elif "flexible fuel" in s or "flex fuel" in s:
        return "Flexible Fuel"
    else:
        return "Unknown"

def extract_horsepower(val):
    if not isinstance(val, str):
        return np.nan
    match = re.search(r'([\d.]+)\s*HP', val, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

def extract_engine_size(val):
    if not isinstance(val, str):
        return np.nan
    match = re.search(r'([\d.]+)\s*L', val, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

def extract_cylinder(val):
    if not isinstance(val, str):
        return np.nan
    match = re.search(r'\bV?(\d+)\s*(Cylinder|V\d|I\d|Rotary)', val, re.IGNORECASE)
    return int(match.group(1)) if match else np.nan

def extract_is_electric(val):
    if not isinstance(val, str):
        return 0
    return 1 if re.search(r'electric\s+motor', val, re.IGNORECASE) else 0

def extract_is_turbo(val):
    if not isinstance(val, str):
        return 0
    return 1 if re.search(r'turbo|supercharged', val, re.IGNORECASE) else 0

def extract_fuel_system(val):
    if not isinstance(val, str):
        return np.nan
    match = re.search(r'([A-Za-z\s]+Fuel\s*System|[A-Za-z\s]+Fuel)$', val.strip(), re.IGNORECASE)
    return match.group(1).strip() if match else "Unknown"

def extract_info(X, engine_col="engine"):
  X = X.copy()
  specs_df = pd.DataFrame({
        "horsepower": X[engine_col].apply(extract_horsepower),
        "engine_size": X[engine_col].apply(extract_engine_size),
        "cylinder": X[engine_col].apply(extract_cylinder),
        "is_electric": X[engine_col].apply(extract_is_electric),
        "is_turbo": X[engine_col].apply(extract_is_turbo),
        "fuel_system": X[engine_col].apply(extract_fuel_system),
    })

  fuel = X['engine'].apply(extract_fuel)
  X['fuel_type'] = X['fuel_type'].mask(X['fuel_type'] == "Unknown", fuel)

    # Set 0 untuk EV jika hp/cylinder/size NaN
  mask_ev = specs_df["is_electric"] == 1
  specs_df.loc[mask_ev & specs_df["horsepower"].isna(), "horsepower"] = 0
  specs_df.loc[mask_ev & specs_df["cylinder"].isna(), "cylinder"] = 0
  specs_df.loc[mask_ev & specs_df["engine_size"].isna(), "engine_size"] = 0

  X.drop(['engine','model'], axis = 1,inplace=True)

  return pd.concat([X, specs_df], axis=1)

def imputer_transform(data ,imp):
  data = data.copy()
  # Non-electric impute with median
  cols=["horsepower", "engine_size", "cylinder"]
  mask_non_ev = data['is_electric'] == 0
  data.loc[mask_non_ev, cols] = imp.transform(data.loc[mask_non_ev, cols])

  # Impute accidend and clean_title
  data['accident'] = data['accident'].fillna('Unknown')
  data['clean_title'] = data['clean_title'].fillna('Unknown')

  return data



def simplify_color(color):
    """
    Menyederhanakan kategori warna menjadi:
    blue, red, black, silver, white, gold, orange, purple, beige, other, unknown
    """
    if not isinstance(color, str) or color.strip() == "" or color.strip() =="-":
        return "Unknown"

    color_lower = color.lower()

    mapping = {
        "blue": ["blue", "navy", "aqua", "turquoise", "teal"],
        "red": ["red", "maroon", "burgundy"],
        "black": ["black", "ebony", "onyx"],
        "silver": ["silver", "gray", "grey", "graphite", "charcoal"],
        "white": ["white", "ivory", "cream", "pearl"],
        "gold": ["gold", "champagne"],
        "orange": ["orange", "copper", "bronze"],
        "purple": ["purple", "violet", "plum", "lavender"],
        "beige": ["beige", "tan", "sand", "khaki", "camel"]
    }

    for base_color, keywords in mapping.items():
        if any(kw in color_lower for kw in keywords):
            return base_color

    return "Other"

def color_transform(X):
  X["ext_col"] = X["ext_col"].apply(simplify_color)
  X["int_col"] = X["int_col"].apply(simplify_color)
  return X

def scaler_transform(df, scaler):
  cols = ['model_year','milage','horsepower','engine_size','cylinder']
  df[cols] =df[cols].astype(float)
  df.loc[:, cols] = scaler.transform(df[cols]).astype(float)
  return df

def encoder_transform(X, encoder):
  cat_cols = ['brand','fuel_type','transmission','ext_col','int_col','accident', 'clean_title','fuel_system']
  encoded = encoder.transform(X[cat_cols])
  df_encoded = pd.concat([X.drop(columns=cat_cols), encoded], axis=1)
  return df_encoded


def preprocessing_pipeline(X, imputer, scaler, encoder):
  X_clean = X.copy()
  X_clean = format_cleaning(X_clean)
  X_clean = extract_info(X_clean)
  X_clean = imputer_transform(X_clean, imputer)
  X_clean = color_transform(X_clean)
  X_clean = scaler_transform(X_clean, scaler)
  X_clean = encoder_transform(X_clean, encoder)
  return X_clean