import pandas as pd
import numpy as np
from statistics import mean
from surprise import Reader, Dataset, BaselineOnly
from surprise.model_selection.validation import cross_validate
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
def Save_Object(obj,filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    return

def Load_Object(filename):
    with open(filename, 'rb') as file:
        obj= pickle.load(file)
    return obj

# 1. Read data
data1 = pd.read_csv("Review_new.csv", encoding='latin-1')
data2 = pd.read_csv("Product_new.csv", encoding='latin-1')

#--------------
# GUI
st.title("Data Science Project")
st.write("## Recommender System for TIKI products")

# 2. Data pre-processing

# 3. Build model
# Collaborative filtering
pkl_filename = "recommendation_tiki.pkl"
with open(pkl_filename, 'rb') as file:  
    CF_model = pickle.load(file)

# Content-based filtering: Cosine Similarities
CB_df = pd.read_csv("CB_cosinesimilarity.csv")

# 4. Evaluate model
result = pd.read_csv("evaluation.csv")

#5. Prediction
# Collaborative filtering
def recommendation_cf (customerid, algorithm, df):
    df_score = df[['product_id']]
    df_score['EstimateScore'] = df_score['product_id'].apply(lambda x: algorithm.predict(customerid, x).est) 
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates()
    df_score = df_score[df_score['EstimateScore']>=3][:5]
    return df_score

# Content-based filtering
# Get product information
def recommendation_cbf (customerid, algorithm, df):
    df_score = df[['product_id']]
    df_score['EstimateScore'] = df_score['product_id'].apply(lambda x: algorithm.predict(customerid, x).est) 
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates()
    df_score = df_score[df_score['EstimateScore']>=3][:5]
    return df_score


#6. Save models
#7. Load models 


# GUI
menu = ["About project", "Business Objective","Build Project","New Prediction"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "About project":
    st.write("""#### 1. Concept""")
    st.write("""V??? b???n ch???t ch??ng ta c?? th??? coi m???t h??? th???ng g???i ?? gi???ng nh?? m???t ng?????i mai m???i. N?? d??? ??o??n s??? th??ch c???a ch??ng ta v?? t??m ki???m c??c ?????i t?????ng ti???m n??ng ph?? h???p v???i s??? th??ch ???? ????? g???i ?? cho ch??ng ta. Gi??? s??? b???n l?? m???t g?? ?????c th??n vui t??nh, b???n r???t th??ch nh???ng c?? g??i xinh x???n v?? n??ng ?????ng, ho???t b??t. Nh??ng th??? t?????ng t?????ng m?? xem, s???c ng?????i c?? h???n v?? b???n th?? ho??n to??n kh??ng th??? n??o bi???t h???t m???t ngang m??i d???c c??c m??? nh??n trong thi??n h???. L??c n??y b???n r???t c???n ?????n m???t h??? g???i ?? nh?? th??? m???t b?? m???i cho b???n v???y. D??? hi???u h??n m???t ch??t r???i ph???i kh??ng, b??y gi??? ch??ng ta s??? tr??? l???i m???t v?? d??? ??p d???ng trong m???t l??nh v???c kh?? g???n g??i v???i d??n IT ch??ng ta ???? l?? th????ng m???i ??i???n t???.""")
    st.write("""#### 2. Approaches""")
    st.write("""**The Collaborative filtering** t???c l?? h??? th???ng s??? ph??n t??ch c??c user c?? c??ng ????nh gi??, c??ng mua m???c tin hi???n t???i. Sau ???? t??m ra danh s??ch c??c m???c tin kh??c c??ng ???????c ????nh g??a b???i c??c user n??y, x???p h???ng v?? g???i ?? cho ng?????i d??ng. T?? t?????ng c???a ph????ng ph??p n??y ch??nh l?? d???a tr??n s??? t????ng ?????ng v??? s??? th??ch gi???a c??c ng?????i d??ng ????? ????a ra c??c g???i ??.""")
    st.write("""**The content-based filtering** t???c l?? h??? th???ng s??? quan t??m ?????n n???i dung, ?????c ??i???m c???a m???c tin hi???n t???i v?? sau ???? g???i ?? cho ng?????i d??ng c??c m???c tin t????ng t???. ???? ch??nh l?? tr?????ng h???p th??? nh???t.""")
    col1, col2 = st.columns(2)
    col1.image("collaborative-filtering.png")
    col2.image("content-based-filtering-01.png")


elif choice == "Business Objective":
    st.subheader("Business Objective")
    st.write("""
    ###### Tiki.vn Tiki l?? ???ng d???ng mua s???m shopping ti???n l???i v???i nhi???u m???t h??ng, gian h??ng ?????n t??? nhi???u th????ng hi???u uy t??n. ?????n v???i Tiki, b???n d??? d??ng t??m mua ???????c s???n ph???m ??ang c???n. ????y l?? ???ng d???ng mi???n ph?? tr??n nhi???u n???n t???ng v?? thi???t b???.

    Tiki.vn cung c???p h??n 10 tri???u s???n ph???m t??? 16 ng??nh h??ng l???n, ph???c v??? h??ng tri???u kh??ch h??ng tr??n to??n qu???c, ?????ng th???i ra m???t d???ch v??? TikiNow, giao h??ng nhanh trong 2h. ????y l?? c?? h???i kh??ng th??? t???t h??n ????? c??c Nh?? B??n m??? r???ng k??nh b??n h??ng v?? gia t??ng thu nh???p tr??n s??n Tiki.
    """)  
    st.image("logo-tiki.jpeg")

elif choice == "Build Project":
    st.subheader("Build Project")
    st.write("#### 1. A brief overview of data")
    st.write("##### Dataset - Reviews")
    col1, col2 = st.columns(2)
    col1.metric(label="Rows", value=data1.shape[0])
    col2.metric(label="Columns", value=data1.shape[1])
    st.table(data1[['customer_id','product_id','rating']].head(3))
    st.write("##### Dataset - Products")
    col1, col2 = st.columns(2)
    col1.metric(label="Rows", value=data2.shape[0])
    col2.metric(label="Columns", value=data2.shape[1])
    st.table(data2[['item_id','rating','price','brand']].head(3))

    st.write("#### 2. Exploratory Data Analysis")
    st.write("""##### About Price""")
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    data2.price.plot(kind='box', ax=ax[0])
    data2.price.plot(kind='hist', bins=20, ax=ax[1])
    st.pyplot(fig)
    st.write("""> ###### Comments
                >   
                > - Data has many outliers.
                > - The price has range from more than 0 to 600000.""")
    st.write("""""")

    st.write("""##### The number of product items by brand""")
    brands = data2.groupby('brand')['item_id'].count().sort_values(ascending=False)
    st.bar_chart(brands[1:11])
    st.write("""> ###### Comments
                >   
                > - Top 10 brands had the most product: Samsung, LG, Panasonic, Sony, Yoosee, SanDisk, Apple, URGREEN, TP-Link, Logitech.
                > - In there, Samsung is a brand with the highest number of products. 
                >- Some brands like LG, Panasonic, Sony had the similarity number of product.""")
    st.write("""""")

    st.write("""##### Average price by brand""")
    price_by_brand = data2.groupby(by="brand").mean()["price"]
    st.bar_chart(price_by_brand.sort_values(ascending=False)[:10])
    st.write("""> ###### Comments
                >
                > - Brand Hitachi has the highest average price.
                > - After brand Hitachi, there are some typical brands such as: Surface, Bosch, Black Shark, Apple.""")
    st.write("""""")

    st.write("""##### Product Rating""")
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    data2[['rating']].plot(kind='box', ax=ax[0])
    data2[['rating']].plot(kind='hist', ax=ax[1])
    st.pyplot(fig)
    st.write("""> ###### Comments
                > 
                > - The level of rating is in range from 2 to 5 stars. However, there are some outliers with the level of rating under 2 stars.
                > - The level of rating is mostly in range from 4 to 5 stars.
                > - There are still products which have the rating equals to 0.""")
    st.write("""""")

    st.write("""##### Product Rating & Average Rating""")
    avg_rating_customer = data1.groupby(by="product_id").mean()["rating"].to_frame().reset_index()
    avg_rating_customer = avg_rating_customer.rename({'rating':'avg_rating'}, axis=1)
    products = data2.merge(avg_rating_customer, left_on="item_id", right_on="product_id", how="left")
    fig1, ax = plt.subplots(1,2, figsize=(10,6))
    sns.histplot(products, x='rating', ax=ax[0])
    sns.histplot(products, x='avg_rating', ax=ax[1])
    st.pyplot(fig1)
    st.write("""> ###### Comments
                >  
                > - The average rating from customers is bigger than 0
                > - Some products have the rating that equals to 0 because of lacking data.""")
    st.write("""""")

    st.write("""##### About Reviews""")
    fig, ax = plt.subplots()
    data1.rating.plot(kind='hist')
    st.pyplot(fig)
    st.write("""> ###### Comments
                > 
                > - Most of products have the rating equals to 5 => products are reviewed positively""")
    st.write("""""")

    st.write("""##### Top 20 customers who reviewed the most""")
    top_rating_customer = data1.groupby('customer_id').count()['product_id'].sort_values(ascending=False)[:20]
    fig, ax = plt.subplots()
    ax.bar(x=[str(x) for x in top_rating_customer.index], height=top_rating_customer.values)
    xlabels = top_rating_customer.index
    ax.set_xticklabels(xlabels, rotation=70)
    st.pyplot(fig)
    st.write("""> ###### Comments
                >  
                > - Customer with id = 7737978 conducted reviewing the most.
                > - Other customers are likely to be the same followed by each group.""")

    st.write("#### 3. Build model")
    st.write("#### 4. Evaluation")
    st.code("Test RMSE: "+ str(round(result.iloc[1]['test_rmse'],2)))
    st.code("Test MAE: "+ str(round(result.iloc[1]['test_mae'],2)))
    st.code("Fit Time: "+ str(result.iloc[1]['fit_time']))
    st.code("Test Time: "+ str(result.iloc[1]['test_time']))

    st.write("#### 5. Summary")
    st.info("##### This model is good enough to build Recommender System for Tiki products.")

elif choice == "New Prediction":
    st.subheader("Make New Prediction")
    type = st.radio("Which types of recommendation do you want?", options=("Collaborative Filtering","Content-based Filtering"))
    if type == "Collaborative Filtering":
        customerid = st.text_input("Please enter customer_id")
        clicked = st.button("Show suggestion")
        if customerid or clicked:
            df_new = recommendation_cf(int(customerid), CF_model, data1)
            df_new = df_new.reset_index()
            st.write("""##### List of product suggestion""")            
            st.table(df_new[['product_id']])
    
    if type == "Content-based Filtering":
        productids = st.multiselect("Please choose the product_ids you want", data2['item_id'][:20])
        clicked = st.button("Show suggestion")
        if productids or clicked:
            st.write("""##### List of product suggestion""")
            for productid in productids:
                df_cbf = CB_df[CB_df['product_id']==int(productid)] 
                df_cbf = df_cbf.rename(columns={'rcmd_product_id': 'recommendation_product_id'})
                df_cbf = df_cbf.reset_index(drop=True)                      
                st.table(df_cbf[['recommendation_product_id']][:5])
 
            

    