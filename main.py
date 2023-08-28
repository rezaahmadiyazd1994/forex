from asyncio.windows_events import NULL
from contextlib import nullcontext
from ftplib import parse150
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.corpus import stopwords
import os
from datetime import datetime

global signal_news
signal_news = ""

class Data:
    def load_data(self,csv_path):
        global df_final
        df_final = pd.read_csv(csv_path)
   
    def data_preprocessing(self,drop_list):
        # Data pre-processing
        X = df_final.drop(drop_list,axis=1).values
        y = df_final['Action'].values

        # Split Train And Test Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Scale And Standard Variables
        global sc
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    def load_model(self,path_json,path_h5):
        global loaded_model
        # load json and create model
        json_file = open(path_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(path_h5)
    
    def live_data(self,urn,api_key):
        global price,change,open_price,high_price,low_price

        # get stream live data
        url = urn
        headers = {'x-access-token': api_key}
        resp = requests.get(url, headers=headers)
        resp_dict = resp.json()

        price = resp_dict.get('price')
        price = float(price)
        self.price = price

        change = resp_dict.get('ch')
        change = float(change)
        self.change = change

        open_price = resp_dict.get('open_price')
        open_price = float(open_price)
        self.open_price = open_price

        high_price = resp_dict.get('high_price')
        high_price = float(high_price)
        self.high_price = high_price

        low_price = resp_dict.get('low_price')
        low_price = float(low_price)
        self.low_price = low_price



    def get_prices(self):
        global prev_open,prev_high,prev_low,prev_close,prev_change,prev_high_low,prev_close_high,prev_sum_4_price,prev_volume,prev_high_open,prev_open_low
        # load prev prices value
        # read file
        with open('price.json', 'r') as myfile:
            data = myfile.read()

        # parse file
        y = json.loads(data)

        #for all data
        prev_open = y["p_open"]
        prev_high = y["p_high"]
        prev_low = y["p_low"]
        prev_close = y["p_close"]
        prev_change = y["p_change"]
        prev_high_low = prev_high - prev_low
        prev_close_high = prev_close - prev_high
        prev_sum_4_price = prev_open + prev_high + prev_low + prev_close

        #for only data1
        prev_volume = y["p_volume"]

        #for only data2
        prev_high_open = prev_high - prev_open

        #for only data3
        prev_open_low = prev_open - prev_low

    def pred(self,fo):
        global pred,signal_data,buy_counter,sell_counter
        new_pred = loaded_model.predict(sc.transform(np.array([[prev_open,prev_high,prev_low,prev_close,fo,prev_change,prev_high_low,prev_close_high,prev_sum_4_price]])))
        pred = new_pred
        new_pred = (new_pred > 0.5)

        sell_counter = 0
        buy_counter = 0

        if (new_pred):
            buy_counter = buy_counter + 1
            signal_data = "Buy"
        else:
            sell_counter = sell_counter + 1
            signal_data = "Sell"

class News:
    global s,count_positive,count_neutral,count_negative,urls,pn,sell_counter,pns,percent_negative,percent_positive
    percent_positive = 0
    percent_negative = 0
    pns = ""

    def ProcessNews(self,urls,element,class_element):
        global pn,pns,signal_news,stop_words
        stop_words = set(stopwords.words('english'))

        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            headlines = soup.find_all(element, class_ = class_element)
            for headline in headlines:
                headline = str(headline)

                word_tokens = word_tokenize(headline)
                filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
                filtered_sentence = []
                        
                for w in word_tokens:
                    if w not in stop_words:
                        filtered_sentence.append(w)
                        
                    lemma_word = []
                    wordnet_lemmatizer = WordNetLemmatizer()
                    for w in filtered_sentence:
                        word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
                        word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
                        word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
                        lemma_word.append(word3)
                
                s = filtered_sentence
            
                tweet = s
                tweet = str(tweet)
                tweet = BeautifulSoup(tweet, "lxml").text

                # create TextBlob object of passed tweet text 
                import re 
                analysis = TextBlob(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())) 
                
                count_negative = 0
                count_positive = 0
                count_neutral = 0

                # set sentiment 
                if(analysis.sentiment.polarity > 0): 
                    count_positive = count_positive + 1
                elif(analysis.sentiment.polarity == 0): 
                    count_neutral = count_neutral + 1
                elif(analysis.sentiment.polarity < 0): 
                    count_negative = count_negative + 1

        try:
            count_all = count_positive + count_negative
            percent_positive = (count_positive * 100 / count_all) - 25
            percent_negative = (count_negative * 100 / count_all) + 25
        except:
            count_all = 0
            percent_positive = 0
            percent_negative = 0

        if(percent_positive > 50):
            buy_counter = buy_counter + 1
            pn = 1
            pns = "Buy"
            signal_news = "Buy"
        elif(percent_negative >= 50):
            sell_counter = sell_counter + 1
            pn = -1
            pns = "Sell"
            signal_news = "Sell"
        else:
            pn = 0
            pns = "Neutral"
            signal_news = "Neutral"

class Final_Calc:

    def save_singal(self):
        global data_class,final_signal
        data_class = Data()
        
        if(buy_counter > sell_counter):
            final_signal = "Buy"
        else:
            final_signal = "Sell"

        # get day
        today = datetime.now()
        day = today.strftime("%d")

        day = str(day)
        datet = str(today.year)
        datem = str(today.month)
        timeh = str(today.hour)
        timem = str(today.minute)
        # Save in text file

        # Directory
        directory = day+"-"+datem+"-"+datet
        
        # Parent Directory path
        parent_dir = "date"
        
        # Path
        path = os.path.join(parent_dir, directory)

        if(os.path.isdir(path)): 
            a = 1
        else:
            os.mkdir(path)

        new_path = timeh + "-" + timem + ".txt"
        pred = open("date/"+directory+"/"+new_path,'w')
        txt = 'Data: ' + signal_data + '\nNews: ' + signal_news + '\nAll: ' + final_signal
        pred.write(txt)

        global sp
        if(signal_news == 'Sell' or signal_news == 'Buy'):
	        sp = "          "
        else:
	        sp = "       "

        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")

        print("      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      ")
        print("      ▓                                                                                                              ▓      ")
        print("      ▓                                XAUUSD (Gold Price in US Dollars) Prediction                                  ▓      ")
        print("      ▓                                                                                                              ▓      ")

        print("      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      ")

        print("")
        print("        Open	      High	    Low	        Price          Change      	 Data	        News           Final   ")

        print("      ────────────────────────────────────────────────────────────────────────────────────────────────────────────────      ")
        print("     ",open_price,"	    ",high_price," 	 ",low_price,"     ",price,"      ",change,"  	        ",signal_data,"	       ",signal_news,sp,final_signal,"    ")
        print("")
        print("      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      ")

        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")      

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------End Code----------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------


def pred_gold():
    global gold,gold_news,gold_final,gold_news_1,gold_news_2,gold1,gold2,gold3
    gold = Data()
    gold1 = Data()
    gold2 = Data()
    gold3 = Data()
    gold_news = News()
    gold_news_1 = News()
    gold_news_2 = News()
    gold_final = Final_Calc()

    # load gold datas
    gold1.load_data('data/1/data.csv')
    gold2.load_data('data/2/data.csv')
    gold3.load_data('data/3/data.csv')

    # drop not needed features from cvs file list
    gold1.drop_list = ['Date','Action','Change','Open','High','Low','Close','Volume']
    gold1.data_preprocessing(gold1.drop_list)

    gold2.drop_list = ['Date','Action','Change','Open','High','Low','Close','Volume']

    gold2.data_preprocessing(gold2.drop_list)

    gold3.data_preprocessing(gold2.drop_list)

    # load gold model and weight of model
    gold1.load_model('model/model-1/model.json','model/model-1/model.h5')
    gold2.load_model('model/model-2/model.json','model/model-2/model.h5')
    gold3.load_model('model/model-3/model.json','model/model-3/model.h5')

    # get live stream gold data from goldapi.io
    gold.live_data('http://www.goldapi.io/api/XAU/USD','goldapi-19j4clrlk5p94q2-io')

    # get gold prices from json file
    gold.get_prices()

    # prediction
    gold1.pred(prev_volume)
    gold2.pred(prev_high_open)
    gold3.pred(prev_open_low)

    # search in search engine
    # urls 
    urls1 = [
    'https://gerdoo.me/search/?query=xauusd%20twitter%20buy%20sell&page=5&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20buy%20sell&page=4&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20buy%20sell&page=3&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20buy%20sell&page=2&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20buy%20sell&page=1&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20analysis&page=5&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20analysis&page=4&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20analysis&page=3&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20analysis&page=2&date=day',
    'https://gerdoo.me/search/?query=xauusd%20twitter%20analysis&page=1&date=day',
    'https://gerdoo.me/search/?query=political%20news&page=5&date=day',
    'https://gerdoo.me/search/?query=political%20news&page=4&date=day',
    'https://gerdoo.me/search/?query=political%20news&page=3&date=day',
    'https://gerdoo.me/search/?query=political%20news&page=2&date=day',
    'https://gerdoo.me/search/?query=political%20news&page=1&date=day',
    'https://gerdoo.me/search/?query=Political%20and%20economic%20news&page=5&date=day',
    'https://gerdoo.me/search/?query=Political%20and%20economic%20news&page=4&date=day',
    'https://gerdoo.me/search/?query=Political%20and%20economic%20news&page=3&date=day',
    'https://gerdoo.me/search/?query=Political%20and%20economic%20news&page=2&date=day',
    'https://gerdoo.me/search/?query=Political%20and%20economic%20news&page=1&date=day',
    ]


    # process news 1
    gold_news_1.ProcessNews(urls1,'span','highlight-text')

    # url2
    urls2 = [
    'https://www.tradingview.com/symbols/XAUUSD/news/?exchange=FX'
    ]

    gold_news_2.ProcessNews(urls2,'span','title-rY32JioV')

    #compire news with data
    gold_final.save_singal()


#Run
pred_gold()

