import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model = pickle.load(open('model.pkl','rb'))
vect = pickle.load(open('vector.pkl','rb'))

ps = PorterStemmer()

def stemming(news):
    news = re.sub('[^a-zA-Z]', ' ',news)
    news = news.lower()
    news = news.split()
    news = [ps.stem(word) for word in news if word not in stopwords.words('english')]
    news = ' '.join(news)
    return news

def fakenews(news):
    news = stemming(news)
    input_data = [news]
    vectors = vect.transform(input_data)
    prediction = model.predict(vectors)
    return prediction

if __name__ == '__main__':
    st.title('Fake News Detection App')
    st.subheader('Input the news content below')
    sentence = st.text_area('Enter the news here',height=200)
    predict_btt = st.button('Predict')
    if predict_btt:
        predict_case = fakenews(sentence)
        print(predict_case)
        if predict_case == [0]:
            st.success('Reliable')
        if predict_case == [1]:
            st.warning('Unreliable')




