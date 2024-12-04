# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image 
import cv2
import math
import numpy as np
import io
import seaborn as sns
import gradio as gr
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import plotly.express as px
import warnings
import zipfile
warnings.filterwarnings('ignore')
print("enddddd")


# Load transaction data frame
transactions = pd.read_csv('transactions_train.csv')
transactions = transactions.head(2000)

transactions['bought'] = 1 #the interaction matrix will be binary
df=transactions[['customer_id', 'article_id', 'bought']]


df = df.drop_duplicates()

# Creating a sparse pivot table with customers in rows and items in columns
customer_items_matrix_df = df.pivot(index   = 'customer_id', 
                                    columns = 'article_id', 
                                    values  = 'bought').fillna(0)

customer_items_matrix_df_train, customer_items_matrix_df_test = train_test_split(customer_items_matrix_df,test_size=0.33, random_state=42)


def autoEncoder(X):
    '''
    Autoencoder for Collaborative Filter Model
    '''

    # Input
    input_layer = Input(shape=(X.shape[1],), name='UserScore')
    
    # Encoder
    enc = Dense(512, activation='selu', name='EncLayer1')(input_layer)

    # Latent Space
    lat_space = Dense(256, activation='selu', name='LatentSpace')(enc)
    lat_space = Dropout(0.8, name='Dropout')(lat_space) # Dropout

    # Decoder
    dec = Dense(512, activation='selu', name='DecLayer1')(lat_space)

    # Output
    output_layer = Dense(X.shape[1], activation='linear', name='UserScorePred')(dec)

    # this model maps an input to its reconstruction
    model = Model(input_layer, output_layer)    
    
    return model


# We want to map the input to itself
X = customer_items_matrix_df_train.values

# Build model
model = autoEncoder(X)

model.compile(optimizer = Adam(learning_rate=0.0001), loss='mse')

model.fit(x=X, y=X,
    epochs=60,# Using 50 here instead of 10 or 20 improved the performance very much!
    batch_size=16,
    shuffle=True,
    validation_split=0.1)


# Predict new Matrix Interactions, set score zero on articles customer haven't purchased
new_matrix = model.predict(X) * (X[0] == 0)


# converting the reconstructed matrix back to a Pandas dataframe
new_customer_items_matrix_df  = pd.DataFrame(new_matrix, 
                                            columns = customer_items_matrix_df_train.columns, 
                                            index   = customer_items_matrix_df_train.index)


def recommender_for_customer(customer_id, interact_matrix, df_content, topn = 5):
    '''
    Recommender Articles for Customers
    '''
    pred_scores = interact_matrix.loc[customer_id].values

    df_scores   = pd.DataFrame({'article_id': list(customer_items_matrix_df.columns), 
                               'score': pred_scores})

    df_rec      = df_scores.set_index('article_id')\
                    .join(df_content.set_index('article_id'))\
                    .sort_values('score', ascending=False)\
                    .head(topn)[['score', 'prod_name']]
    
    return df_rec[df_rec.score > 0]

articles = pd.read_csv("articles.csv")


def display_items(item_ids, title, image_folder='https://drive.google.com/drive/folders/1aQjL2NPuREcnueljBliPTTePCVUc32es?usp=sharing'):
    # Adjust size to make the visualizations bigger
    k = len(item_ids)
    plt.close('all')
    fig_width = max(20, 5 * k)  # Scale width based on the number of items
    fig_height = 10  # Fixed height for better visibility

    fig = plt.figure(figsize=(fig_width, fig_height))  # Adjust figure size here
    plt.title(title, size=24)  # Increase the title size
    plt.axis('off')

    for item, i in zip(item_ids, range(1, k + 1)):
        article_id = str(item)
        path = os.path.join(image_folder, f"0{article_id[:2]}", f"0{article_id}.jpg")
        if os.path.exists(path):
            image = plt.imread(path)
            fig.add_subplot(1, k, i)
            plt.title(f"Article {item}", size=16)  # Increase subtitle size
            plt.axis('off')
            plt.imshow(image)
            plt.show()
        else:
            print(f"Image not found: {path}")
            fig.add_subplot(1, k, i)
            plt.title(f"Article {item} (Missing)", size=16)
            plt.axis('off')

    # Save plot to buffer for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')  # Ensure tight bounding box
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf)

def fetch_and_recommend(customer_id):
    # Fetch previously purchased items
    hist_customer = recommender_for_customer(
        customer_id=customer_id, 
        interact_matrix=customer_items_matrix_df, 
        df_content=articles
    )
    prev_items = hist_customer.index.values
    print("pre items",prev_items)
    prev_plot = display_items(prev_items, title="Products Previously Purchased")

    # Fetch recommended items
    hist_customer_rec = recommender_for_customer(
        customer_id=customer_id, 
        interact_matrix=new_customer_items_matrix_df, 
        df_content=articles
    )
    next_items = hist_customer_rec.index.values

   

    next_plot = display_items(next_items, title="Recommended Products")

    # Return plots
    return prev_plot, next_plot

interface = gr.Interface(
    fn=fetch_and_recommend,
    inputs=gr.Textbox(label="Enter Customer ID"),
    outputs=[gr.Image(label="Previously Purchased"), gr.Image(label="Recommended Products")],
    title="H&M Fashion Recommendation",
    description="Input a customer ID to view their purchase history and recommendations."
)

interface.launch()


