from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    
    # Build chained labels for multi-output chained classification.
    from preprocess import build_chained_labels
    df = build_chained_labels(df)
    
    # Ensure text columns have proper type.
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype('U')
    
    # Group by the column defined in Config (usually y1).
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print("Processing group:", name)
        X = get_tfidf_embd(group_df)
        data = get_data_object(X, group_df) # Data object now contains chained labels.  
        perform_modelling(data, group_df, name)
        print("Completed processing group:", name)