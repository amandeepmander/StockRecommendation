# -*- coding: utf-8 -*-

from scripts import tabledef
from scripts import forms
from scripts import helpers
from flask import Flask, redirect, url_for, render_template, request, session
import json
import sys
import os
import stripe

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import urllib.request
import datetime as dt
import tensorflow as tf

app = Flask(__name__)
app.secret_key = os.urandom(12)  # Generic key for dev purposes only

stripe_keys = {
  'secret_key': os.environ['STRIPE_SECRET_KEY'],
  'publishable_key': os.environ['STRIPE_PUBLISHABLE_KEY']
}
stripe.api_key = stripe_keys['secret_key']

# Alpha Vantage API key
api_key = 'A3Y10PJZ4M4FYKSW'

# Model files
model_AAPL = 'AAPL_model.sav'

# Heroku
#from flask_heroku import Heroku
#heroku = Heroku(app)

# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    user = helpers.get_user()
    return render_template('home.html', user=user, key=stripe_keys['publishable_key'])


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))

@app.route('/charge', methods=['POST'])
def charge():

    # amount in cents
    amount = 500

    customer = stripe.Customer.create(
        email='sample@customer.com',
        source=request.form['stripeToken']
    )

    stripe.Charge.create(
        customer=customer.id,
        amount=amount,
        currency='usd',
        description='Flask Charge'
    )

    return render_template('charge.html', amount=amount)

@app.route('/predict',methods=['POST'])
def predict():


    if request.method == 'POST':

        ticker = request.form['ticker']
        # Get daily adjusted stock price data
        url_string_adjdaily = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=%s&outputsize=full&apikey=%s'%(ticker, api_key)
        with urllib.request.urlopen(url_string_adjdaily) as url:

            data = json.loads(url.read().decode())

            # Extract data
            data = data['Time Series (Daily)']
            df_adjclose = pd.DataFrame(columns=['Date','AdjClose'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['5. adjusted close'])]
                df_adjclose.loc[-1,:] = data_row
                df_adjclose.index = df_adjclose.index + 1

        historical_days = 100

        # Take last x number of data points
        df_adjclose_pred = df_adjclose.head(historical_days)

        # Drop date variable
        df_final_pred = df_adjclose_pred.drop(['Date'], 1)

        # Reverse the dataframe to get data in chronological order
        df_final_pred = df_final_pred.iloc[::-1]

        # fix random seed for reproducibility
        np.random.seed(7)

        # load the dataset
        dataframe_pred = df_final_pred['AdjClose']
        dataset_pred = dataframe_pred.values
        dataset_pred = dataset_pred.astype('float32')
        dataset_pred = dataset_pred.reshape(1, -1)

        # load the model from disk
        loaded_model = joblib.load(model_AAPL)

        
        # Make prediction
        pred_data = np.reshape(dataset_pred, (dataset_pred.shape[0], 1, dataset_pred.shape[1]))
        my_prediction = loaded_model.predict(pred_data)
        if my_prediction < df_final_pred['AdjClose'].iloc[-1]:
            reco = "Sell"
        else:
            reco = "Buy"

        
        return render_template('result.html',prediction = reco)

# ======== Main ============================================================== #
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
