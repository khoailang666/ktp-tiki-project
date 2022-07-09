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
    st.write("""Về bản chất chúng ta có thể coi một hệ thống gợi ý giống như một người mai mối. Nó dự đoán sở thích của chúng ta và tìm kiếm các đối tượng tiềm năng phù hợp với sở thích đó để gợi ý cho chúng ta. Giả sử bạn là một gã độc thân vui tính, bạn rất thích những cô gái xinh xắn và năng động, hoạt bát. Nhưng thử tưởng tượng mà xem, sức người có hạn và bạn thì hoàn toàn không thể nào biết hết mặt ngang mũi dọc các mỹ nhân trong thiên hạ. Lúc này bạn rất cần đến một hệ gợi ý như thể một bà mối cho bạn vậy. Dễ hiểu hơn một chút rồi phải không, bây giờ chúng ta sẽ trở lại một ví dụ áp dụng trong một lĩnh vực khá gần gũi với dân IT chúng ta đó là thương mại điện tử.""")
    st.write("""#### 2. Approaches""")
    st.write("""**The Collaborative filtering** tức là hệ thống sẽ phân tích các user có cùng đánh giá, cùng mua mục tin hiện tại. Sau đó tìm ra danh sách các mục tin khác cũng được đánh gía bởi các user này, xếp hạng và gợi ý cho người dùng. Tư tưởng của phương pháp này chính là dựa trên sự tương đồng về sở thích giữa các người dùng để đưa ra các gợi ý.""")
    st.write("""**The content-based filtering** tức là hệ thống sẽ quan tâm đến nội dung, đặc điểm của mục tin hiện tại và sau đó gợi ý cho người dùng các mục tin tương tự. Đó chính là trường hợp thứ nhất.""")
    col1, col2 = st.columns(2)
    col1.image("collaborative-filtering.png")
    col2.image("content-based-filtering-01.png")


elif choice == "Business Objective":
    st.subheader("Business Objective")
    st.write("""
    ###### Tiki.vn Tiki là ứng dụng mua sắm shopping tiện lợi với nhiều mặt hàng, gian hàng đến từ nhiều thương hiệu uy tín. Đến với Tiki, bạn dễ dàng tìm mua được sản phẩm đang cần. Đây là ứng dụng miễn phí trên nhiều nền tảng và thiết bị.

    Tiki.vn cung cấp hơn 10 triệu sản phẩm từ 16 ngành hàng lớn, phục vụ hàng triệu khách hàng trên toàn quốc, đồng thời ra mắt dịch vụ TikiNow, giao hàng nhanh trong 2h. Đây là cơ hội không thể tốt hơn để các Nhà Bán mở rộng kênh bán hàng và gia tăng thu nhập trên sàn Tiki.
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
 
            

    