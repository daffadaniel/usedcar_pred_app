import joblib
import pandas as pd

def load_data(fname: str) -> pd.DataFrame:
  """
  Membaca file csv dan mengembailkan DataFrame

  Args: 
      fname (str): path ke file CSV yang akan dibaca.

  Returns:
      data (pd.DataFrame)
  """
  data = pd.read_csv(fname,index_col='id')
  print("Data Shape: ", data.shape)
  return data

def split_input_output(data: pd.DataFrame, target_col: str):
    """
    Split variabel prediktor dengan variabl target 
    
    Args: 
        data (pd.DataFrame): Data yang ingin di split
        target_col (str): variabel target pada data
    
    Returns:
        X (pd.DataFrame): variabel prediktor
        y (pd.Series): variabel target
    """
    X = data.drop(columns = target_col)
    y = data[target_col]
    print(f"Original data shape: {data.shape}") 
    print(f"X data shape: {X.shape}")
    print(f"y datashape: {y.shape}")
    return X,y 


def serialize_data(data: pd.DataFrame, path: str):
    """
     Menyimpan data ke dalam file dengan format pickle (.pkl) menggunakan joblib.

    Args:
        data (pd.DataFrame): Data yang ingin di simpan. 
        path (str): Path lengkap lokasi penyimpanan file, termasuk nama file dan ekstensi (.pkl).

    Returns:
        None
    """
    joblib.dump(data, path)

def deserialize_data(path: str):
    """
    Melakukan deserialisasi (load) data dari file yang disimpan dalam format pickle menggunakan joblib.

    Args:
        path (str): Alamat atau path file tempat data disimpan.

    Returns:
        data (pd.DataFrame): data yang di deserialize
    """
    data =  joblib.load(path)
    return data
