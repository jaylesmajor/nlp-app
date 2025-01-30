![photo_2024-10-13_22-40-15](https://github.com/user-attachments/assets/063ed929-b3cf-4d7b-8e92-3a18bc4e85d8)


# SandBar reviews -  NLP app (Sentiment Analysis) with time series analysis.
1. [Introduction](#introduction-)  
2. [Description](#description-)  
3. [Installation](#installation-)  
4. [Contact](#contact-)  

## Introduction 📝  
This project concerns natural language processing (NLP) for Sandbar, which involves VADER and Naive Bayes sentiment analysis and average star ratings with moving averages. Creating the NLP app also consists of cleaning the text and time series preprocessing for time series analysis. This project or app is useful for businesses or organizations who wish to understand and grasp their performance 

**Technologies and Concepts used**  
1. NLTK 🛠️ for tokenization, stemming, and lemmatization.  
2. Vader and Naive Bayes 💬 for sentiment analysis.  
3. Matplotlib 📊 and Seaborn 🎨: For visualizing text data, word frequencies, and model performance.  
4. Streamlit 🌐 for deployment of the app.  

## Description 🎯  
**EDA** (Below are examples of the EDA performed and its results)  
![rating sand](https://github.com/user-attachments/assets/0c87e568-d2c3-42c2-990a-2a4010f7681b)  

📈 **Distribution of Star Ratings for Sandbar**  
- ⭐️ **5-Star Dominance**: Most reviews are 5-star, indicating high customer satisfaction.  
- ⚠️ **Low Negative Ratings**: 1-star ratings exist but are much less frequent, suggesting fewer negative experiences.  
- ⚖️ **Few Neutral Reviews**: 2- and 3-star ratings are rare, showing a tendency for extreme ratings (positive or negative).  

**Feature Engineering**  
![Feature Engineering](https://github.com/user-attachments/assets/5938adaa-810b-4a2d-9856-74b65d4f98d7)  

🔍 This image highlights the finished result of feature engineering, particularly **Tokenization** and **Lemmatization**.  

**Time Series Analysis**  
![time series](https://github.com/user-attachments/assets/bce7cee1-43ac-45a3-a69f-ee78a39e3837)  

📆 The chart shows an overall upward trend in average star ratings over time, with the 3-month moving average smoothing out monthly fluctuations. Despite some sharp dips, the consistent improvement suggests growing customer satisfaction.

**Sentiment Analysis with VADER and Naive Bayes**  
![VADER1](https://github.com/user-attachments/assets/05e459f3-85ad-4f47-b0f5-676cf4baacaa)


VADER and TF-IDF Analysis:

💡 The VADER model accurately classified 108,655 positive reviews, but misclassified 12,306 negative and 682 neutral reviews as positive. For negative reviews, it correctly identified 14,585 but misclassified 12,306 as positive, while neutral predictions were the least accurate with only 269 correct out of 11,199 total.

![naive bayes](https://github.com/user-attachments/assets/a56b3f93-d275-46ea-a0fa-1f4c12fefb1c)

💡 The Naive Bayes model showed better balance, correctly classifying 22,182 positive reviews with only 82 misclassifications (63 as negative, 19 as neutral). For negative reviews, 2,168 were accurate, with 3,365 misclassified as positive. Neutral reviews were handled more effectively, with 2,179 correct predictions and just 92 errors.



🌐Streamlit Deployment - https://time-series-sentiment-analysis-v1.streamlit.app/



## Installation ⚙️  

**Clone the repo**  
```bash  
git clone https://github.com/jaylesmajor/nlp-app  
```

## Contact 📧
📌 Jayles Escolar - jayles.escolar@yahoo.com

🌐 Project link: https://github.com/jaylesmajor/nlp-app

