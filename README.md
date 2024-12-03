
# H&M Personalized Fashion Recommendations

This project aims to recommend fashion products to customers of H&M using collaborative filtering and content-based techniques. The dataset includes metadata about products, customer information, transaction history, and product images. The application is deployed on Hugging Face for interactive product recommendations.

---

## Dataset Overview

The dataset is sourced from the [H&M Personalized Fashion Recommendations competition on Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data). It consists of:

- **`articles.csv`**: Metadata about products (e.g., product group, garment type, and more).
- **`customers.csv`**: Customer data (e.g., age, membership status, and demographics).
- **`transactions_train.csv`**: Historical transaction data (e.g., price, products purchased, and timestamps).
- **Images Folder**: Contains product images corresponding to the `articles.csv`.

---

## Project Files

### **1. Notebooks**
- **`EDA.ipynb`**: Exploratory Data Analysis (EDA) to understand and visualize the dataset, including metadata, customer insights, and transaction patterns.
- **`Collaborative_filtering.ipynb`**: Implements a collaborative filtering model using an autoencoder to recommend products based on past customer purchases.
- **`fashion_rec.ipynb`**: Uses cosine similarity for content-based recommendations, leveraging product metadata.

### **2. Auto-EDA Reports**
Generated using **Sweetviz**, providing insights into each dataset:
- **`customer_report_AutoEDA.html`**
- **`articles_report_AutoEDA.html`**
- **`transaction_report_AutoEDA.html`**

### **3. Supporting Files**
- **`requirements.txt`**: Lists all the dependencies required to run the project, ensuring compatibility and reproducibility.
- **`file_links.json`**: Maps each product image file name to its corresponding publicly accessible Google Drive link for visualization in the recommendation system.

---

## Interactive Application

The recommendation system is deployed on Hugging Face Spaces. It provides:
1. Visualization of products previously purchased by a customer.
2. Recommendations of new products based on historical purchases.

**Live Demo**: [Hugging Face Deployment](https://huggingface.co/spaces/Rishi3499/DataMining)

---

## Dataset Link

Access the original dataset on Kaggle: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

---

## Setup Instructions

### 1. Clone the Repository
```bash
https://github.com/riship1999/Personalized_Fashion_Recommendation.git
cd Personalized_Fashion_Recommendation
```

### 2. Install Dependencies
Use the provided `requirements.txt` file to set up the environment:
```bash
pip install -r requirements.txt
```

### 3. Add Image Links
Ensure `file_links.json` contains valid Google Drive links to all product images. You can generate this file programmatically if needed.

### 4. Run the Hugging Face Application
Run the `app.py` script locally or deploy it directly on Hugging Face Spaces.

---

## Methodology

### **1. Collaborative Filtering**
A neural network-based autoencoder is trained to find latent features of users and products, generating personalized recommendations.

### **2. Content-Based Filtering**
Cosine similarity is used to recommend products based on similarity in metadata (e.g., garment group, product group).

### **3. Visualization**
Product images are dynamically fetched using Google Drive links, providing a visually intuitive recommendation system.

---

## Contribution

Feel free to contribute by improving the models, optimizing deployment, or enhancing visualization.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
