#Import required libraries
from flask import Flask, render_template, url_for, request
from aitextgen import aitextgen
import datetime as DT
import pandas as pd
import requests,json

#Global Variables
#api_key = '499e16e687954a609ad70fc93e1f6e9e'
api_key = '4e728bc59b5a4eeab9dcad280f0b0bf0'
keywords = ['china','euro','pound','dollar','yen','japan','franc','russia','inflation','monetary','forex','currency']
ai = aitextgen(tf_gpt2="124M")
url = 'https://newsapi.org/v2/everything?q=Apple&from=2022-05-20&sortBy=popularity&apiKey=API_KEY'



app = Flask(__name__)

#Home Page
@app.route("/", methods = ['GET','POST'])
def home():
    dates = ''
    refresh_prompt = ''
    results = ''
    header = 'Current News AI Generator'
    return_string = '''
    Welcome to this Flask App. In this app you will be able to leverage the OpenAI GPT-2 124M model in order to generate up to date summaries of the financial world. To explain the workings behind this model,
    when you click the Refresh button, an API will collect the most recent news headlines from the previous week, filtering for a series of keywords related to the currency markets. The keywords include
    the actual names of currencies themselves (Euro, Pound, Dollar, Yen etc.) and also big countries in order to account for political impacts on the currency markets. Our model will then train on the
    headings of these news articles and return generated text based on what has been provided. Please note that this model takes a while to run and so the results will also take some time to appear.
    '''
    if request.method == 'POST':
        if request.form['refresh'] == 'Refresh':
            refresh_prompt = 'Please wait whilst the model trains...'
            results,date = refresh()
            dates = 'Collecting data from {} to present'.format(date)
    return render_template("index.html", header = header, content = return_string,refresh_prompt = refresh_prompt, results = results, dates = dates)

#Refresh Function
def refresh():
    output = ''
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=7)
    date = str(week_ago.year) + '-' + str(week_ago.month) + '-' + str(week_ago.day)
    train = pd.DataFrame(columns = ['source', 'author', 'title', 'description', 'url', 'urlToImage','publishedAt', 'content'])
    for word in keywords:
        url = 'https://newsapi.org/v2/everything?q={}&from={}&sortBy=popularity&apiKey={}'.format(word, date,api_key)
        text = requests.get(url)
        text = text.json()
        print(text)
        df = pd.DataFrame(text['articles'])
        train = pd.concat([train,df], axis = 0)
    train_string = ""
    for i,r in train.iterrows():
        train_string += r["description"] + ". "
    text_file = open("train.txt", "w")
    n = text_file.write(train_string)
    text_file.close()
    file_path = 'train.txt'
    ai.train(file_path, num_steps = 100)
    output = ai.generate_one(prompt = 'The dollar')
    return output, date

if __name__ == "__main__":
  app.run()
