import json
import os
import pickle
import re
import secrets
from base64 import b64encode, b64decode

import numpy as np
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_wtf.csrf import CSRFProtect
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from astrapy.rest import create_client, http_methods
from dotenv import load_dotenv
import yaml
import logging.config
import logging
from random import randrange
from werkzeug.utils import secure_filename

from form import SingleDataUploadForm, SingleMultipleChoiceForm, MultipleDataUploadForm, DemoForm

# Setting random ID
rand_id = randrange(100000)

# Loading log file configuration
try:
    logging.config.dictConfig(yaml.load(open('logging.conf'), Loader=yaml.FullLoader))
    logging.info(f'user#{rand_id}: Log Configuration Loaded.')
except Exception:
    logging.error(f'user#{rand_id}: Log Configuration Load Error.')

# Load .env file
try:
    load_dotenv()
    logging.info(f'user#{rand_id}: Environment variables Loaded.')
except Exception:
    logging.error(f'user#{rand_id}: Environment variables Loading Error.')

# Initialize Flask
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
logging.info(f'user#{rand_id}: Flask Instantiated.')

# Generate secrets
secret_key = secrets.token_hex(64)
csrf = CSRFProtect()
app.config['SECRET_KEY'] = secret_key

# Initialize App
csrf.init_app(app)
logging.info(f'user#{rand_id}: App Initialized.')

# Load models
if not os.path.exists(os.path.join('models', 'custom', 'custom_model.h5')):
    logging.error(f'user#{rand_id}: Model File Not Found.')
else:
    model = load_model(os.path.join('models', 'custom', 'custom_model.h5'))
    logging.info('Model File Loaded')

# Read Tokenizer
if not os.path.exists(os.path.join('models', 'custom', 'custom_tokenizer.pickle')):
    logging.error(f'user#{rand_id}: Tokenizer File Not Found')
else:
    with open(os.path.join('models', 'custom', 'custom_tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
        logging.info('Tokenizer File Loaded')

# Read Label Encoder
if not os.path.exists(os.path.join('models', 'custom', 'custom_label_encoder.pickle')):
    logging.error(f'user#{rand_id}: Label Encoder File Not Found')
else:
    with open(os.path.join('models', 'custom', 'custom_label_encoder.pickle'), 'rb') as handle:
        label_encoder = pickle.load(handle)
        logging.info('Label Encoder File Loaded')

# Labels
labels = ['Entertainment',
          'Wellness',
          'Politics',
          'Travel',
          'Style & Beauty',
          'Parenting',
          'Healthy Living',
          'Business',
          'Food & Drink',
          'Sports',
          'Comedy',
          'Home & Living',
          'Weddings',
          'Impact',
          'Divorce',
          'Crime',
          'Media',
          'Religion',
          'Arts & Culture',
          'Tech',
          'Style',
          'Science',
          'World News',
          'Money',
          'Arts',
          'Environment',
          'College',
          'Education']

# Astra ------------------------------------------------------------------------
# Setting credentials via environment variables
try:
    ASTRA_DB_ID = os.environ.get('ASTRA_DB_ID')
    ASTRA_DB_REGION = os.environ.get('ASTRA_DB_REGION')
    ASTRA_DB_APPLICATION_TOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')
    ASTRA_DB_KEYSPACE = os.environ.get('ASTRA_DB_KEYSPACE')
    ASTRA_DB_COLLECTION_1 = os.environ.get('ASTRA_DB_COLLECTION_1')
    ASTRA_DB_COLLECTION_2 = os.environ.get('ASTRA_DB_COLLECTION_2')
    astra_client = create_client(astra_database_id=ASTRA_DB_ID,
                                 astra_database_region=ASTRA_DB_REGION,
                                 astra_application_token=ASTRA_DB_APPLICATION_TOKEN)
except Exception:
    logging.error('Environment values retrieval Error')


# -----------------------------------------------------------------------------------
class SingleUpload:
    def __init__(self, text_phrase):
        logging.info(f'user#{rand_id}: Single Form Initialized')
        self.text_phrase = text_phrase

    def single_text_to_list(self):
        logging.info(f'user#{rand_id}: Single Form Word Tokenization Started')
        row_sent = sent_tokenize(self.text_phrase)
        text = []
        for sent in row_sent:
            sent = sent.lower()
            sent = re.sub(r'\W+', ' ', sent)
            words = word_tokenize(sent)
            words = [word for word in words if word not in stopwords.words('english')]
            for word in words:
                text.append(word)
        return [text]

    def process(self):
        logging.info(f'user#{rand_id}: Single Form Processing Started')
        texts_list = self.single_text_to_list()
        # print(texts_list)
        tokenized_text = tokenizer.texts_to_sequences(texts_list)
        # print(tokenized_text)
        return tokenized_text

    @staticmethod
    def predict(tokenized_text, model_name):
        logging.info(f'user#{rand_id}: Single Form Predict Started')
        if model_name == 'custom':
            padded_text = pad_sequences(tokenized_text, maxlen=1720, padding='post')
            predictions = model.predict(padded_text)
            # print(predictions)
            # print((max(predictions[0])))
            # print(np.argmax(predictions[0]))
            return predictions

    @staticmethod
    def metrics(predicted, actual):
        logging.info(f'user#{rand_id}: Single Form Metrics Started')
        conf = confusion_matrix(actual, predicted, labels=list(label_encoder.classes_))
        # print(conf)
        conf = [item_list.tolist() for item_list in conf]
        class_report = classification_report(predicted, actual, output_dict=True)
        accuracy = class_report['accuracy']
        class_report.pop('accuracy')
        metric = {'accuracy': accuracy,
                  'classification_report': class_report,
                  'confusion_matrix': conf}
        # print(type(accuracy))
        # print(type(class_report))
        # print(type(conf[0]))
        # print(type(conf))
        return metric


class MultipleUpload:
    def __init__(self, df_test):
        logging.info(f'user#{rand_id}: Multiple Form Initialized')
        self.df_test = df_test

    def multiple_text_to_list(self):
        logging.info(f'user#{rand_id}: Multiple Form Word Tokenization Started')
        text = []
        for row in self.df_test['Text']:
            row_sent = sent_tokenize(row)
            all_words = []
            for sent in row_sent:
                sent = sent.lower()
                sent = re.sub(r'\W+', ' ', sent)
                words = word_tokenize(sent)
                words = [word for word in words if word not in stopwords.words('english')]
                for word in words:
                    all_words.append(word)
            text.append(all_words)
        return text

    def process(self):
        logging.info(f'user#{rand_id}: Multiple Form Process Started')
        texts_list = self.multiple_text_to_list()
        # print(texts_list)
        tokenized_text = tokenizer.texts_to_sequences(texts_list)
        # print(tokenized_text)
        return tokenized_text

    @staticmethod
    def predict(tokenized_text, model_name):
        logging.info(f'user#{rand_id}: Multiple Form Predict Started')
        if model_name == 'custom':
            padded_text = pad_sequences(tokenized_text, maxlen=1720, padding='post')
            predictions = model.predict(padded_text)
            # print(predictions)
            # print((max(predictions[0])))
            # print(np.argmax(predictions[0]))
            return predictions

    @staticmethod
    def metrics(predicted, actual):
        logging.info(f'user#{rand_id}: Multiple Form Metrics Started')
        conf = confusion_matrix(actual, predicted, labels=list(label_encoder.classes_))
        # print(conf)
        conf = [item_list.tolist() for item_list in conf]
        class_report = classification_report(predicted, actual, output_dict=True)
        accuracy = class_report['accuracy']
        class_report.pop('accuracy')
        metric = {'accuracy': accuracy,
                  'classification_report': class_report,
                  'confusion_matrix': conf}
        # print(type(accuracy))
        # print(type(class_report))
        # print(type(conf[0]))
        # print(type(conf))
        return metric


@app.errorhandler(404)
def page_not_found(error):
    logging.error(f'user#{rand_id}: Error 404 encountered (Not Found) - {error}.')
    return render_template('errors.html', error='404'), 404


@app.errorhandler(403)
def page_not_found(error):
    logging.error(f'user#{rand_id}: Error 403 encountered (Forbidden) - {error}.')
    return render_template('errors.html', error='403'), 403


@app.errorhandler(400)
def page_not_found(error):
    logging.error(f'user#{rand_id}: Error 400 encountered (Bad Request) - {error}.')
    return render_template('errors.html', error='400'), 400


@app.errorhandler(429)
def page_not_found(error):
    logging.error(f'user#{rand_id}: Error 429 encountered (Too Many Requests) - {error}.')
    return render_template('errors.html', error='429'), 429


@app.errorhandler(500)
def page_not_found(error):
    logging.error(f'user#{rand_id}: Error 500 encountered (Internal Server Error) - {error}.')
    return render_template('errors.html', error='500'), 500


@app.errorhandler(502)
def page_not_found(error):
    logging.error(f'user#{rand_id}: Error 502 encountered (Bad Gateway) - {error}.')
    return render_template('errors.html', error='502'), 502


@app.errorhandler(503)
def page_not_found(error):
    logging.error(f'user#{rand_id}: Error 503 encountered (Service Unavailable) - {error}.')
    return render_template('errors.html', error='503'), 503


@app.route('/stat_metrics')
def stat_metrics():
    logging.info(f'user#{rand_id}: Received Data Dictionary.')
    encrypted_data = request.args['data_dict']
    data = json.loads(b64decode(encrypted_data).decode('utf-8'))
    return render_template('stat_metrics.html', table=data)


@app.route('/', methods=['POST', 'GET'])
def home():
    single_form = SingleDataUploadForm()
    single_multiple_choice_form = SingleMultipleChoiceForm()
    multiple_form = MultipleDataUploadForm()
    demo_form = DemoForm()

    # Single Form
    if single_form.validate_on_submit():
        logging.info(f'user#{rand_id}: Single Form Validated Successfully')
        info_dict = {}
        text = single_form.s_input.data
        s_model = single_form.s_model.data
        optional = single_form.s_opt.data
        # print(optional)
        if len(text) < 50:
            logging.error(f'user#{rand_id}: News Article length less than 50')
            flash(f'Length of a news article should be atleast 50 character long', 'danger')
            return redirect(url_for('home'))

        logging.info(f'user#{rand_id}: text data and actual data checked successfully')
        logging.info(f'user#{rand_id}: Length of text data uploaded: {len(text)}')
        logging.info(f'user#{rand_id}: Model selected: {s_model}')
        single_upload = SingleUpload(text)
        tokenized_text = single_upload.process()
        predictions = single_upload.predict(tokenized_text, s_model)
        encoded_value = np.argmax(predictions[0])
        label = label_encoder.inverse_transform([encoded_value])
        info_dict['data'] = [text]
        info_dict['pred'] = label.tolist()
        info_dict['labels'] = list(label_encoder.classes_)
        # Check if optional is there
        if optional != '':  # Optional not empty
            logging.info(f'user#{rand_id}: Actual data selected: {optional}')
            metrics = single_upload.metrics(label.tolist(), [optional])
            info_dict['metrics'] = metrics
            info_dict['actual'] = [optional]
        encrypted_info_dict = json.dumps(info_dict).encode('utf-8')
        flash(f'News Text and Actual Label uploaded successfully', 'success')
        logging.info(f'user#{rand_id}: Single Form Encoded Data Dictionary Sent')
        return redirect(url_for('stat_metrics', data_dict=b64encode(encrypted_info_dict)))

    # Multiple Data Form
    elif multiple_form.validate_on_submit():
        logging.info(f'user#{rand_id}: Multiple Form Validated Successfully')
        info_dict = {}
        df_labels = pd.DataFrame()
        test = multiple_form.m_input.data
        m_model = multiple_form.m_model.data
        actual_labels = multiple_form.m_opt.data
        # print(actual_labels)
        if not secure_filename(test.filename).endswith('.csv'):
            logging.error(f'user#{rand_id}: Wrong file format in {secure_filename(test.filename)}')
            flash(f'Wrong file format in {secure_filename(test.filename)}', 'danger')
            return redirect(url_for('home'))
        if secure_filename(actual_labels.filename) != '' and not secure_filename(actual_labels.filename).endswith('.csv'):  # Optional is not empty
            logging.error(f'user#{rand_id}: Wrong file format in {secure_filename(actual_labels.filename)}')
            flash(f'Wrong file format in {secure_filename(actual_labels.filename)}', 'danger')
            return redirect(url_for('home'))
        if secure_filename(actual_labels.filename) != '' and secure_filename(actual_labels.filename).endswith('.csv'):
            try:
                df_labels = pd.read_csv(actual_labels)
                logging.info(f'user#{rand_id}: {secure_filename(actual_labels.filename)} read')
            except pd.errors.EmptyDataError:
                logging.error(f'user#{rand_id}: Empty labels file uploaded')
                flash(f'File cannot be empty in {secure_filename(actual_labels.filename)}', 'danger')
                return redirect(url_for('home'))

        try:
            df_test = pd.read_csv(test)
            logging.info(f'user#{rand_id}: {secure_filename(test.filename)} read')
        except pd.errors.EmptyDataError:
            logging.error(f'user#{rand_id}: Empty test file uploaded')
            flash(f'File cannot be empty in {secure_filename(test.filename)}', 'danger')
            return redirect(url_for('home'))
        # print(df_test)
        if 'Text' not in df_test.columns.values.tolist():
            logging.error(f"user#{rand_id}: No 'Text' header found in {secure_filename(test.filename)}")
            flash(f"No 'Text' header found in {secure_filename(test.filename)}", 'danger')
            return redirect(url_for('home'))
        if df_test.shape[0] == 0:
            logging.error(f"user#{rand_id}: {secure_filename(test.filename)} has no rows")
            flash(f'{secure_filename(test.filename)} has no rows', 'danger')
            return redirect(url_for('home'))
        if df_test.shape[1] > 1:
            logging.error(f"user#{rand_id}: {secure_filename(test.filename)} has more than one column")
            flash(f'{secure_filename(test.filename)} has more than one column', 'danger')
            return redirect(url_for('home'))
        if df_test.shape[1] == 0:
            logging.error(f"user#{rand_id}: No columns found in {secure_filename(test.filename)}")
            flash(f'No columns found in {secure_filename(test.filename)}', 'danger')
            return redirect(url_for('home'))
        if len(df_test.columns.values.tolist()) == 0:
            logging.error(f"user#{rand_id}: No columns found in {secure_filename(test.filename)}")
            flash(f'No columns found in {secure_filename(test.filename)}', 'danger')
            return redirect(url_for('home'))

        if secure_filename(actual_labels.filename) != '':  # Optional is not empty
            if 'Category' not in df_labels.columns.values.tolist():
                logging.error(f"user#{rand_id}: No 'Category' header found in {secure_filename(actual_labels.filename)}")
                flash(f"No 'Category' header found in {secure_filename(actual_labels.filename)}", 'danger')
                return redirect(url_for('home'))
            category = list(set(df_labels['Category']))
            for category_label in category:
                if category_label not in list(label_encoder.classes_):
                    logging.error(f"user#{rand_id}: Category contains unknown label {category_label}")
                    flash(f'Category contains unknown label {category_label}', 'danger')
                    return redirect(url_for('home'))
            if df_test.shape[0] != df_labels.shape[0]:
                logging.error(f"user#{rand_id}: {secure_filename(test.filename)} and {secure_filename(actual_labels.filename)} doesn't have same no. of rows")
                flash(f"{secure_filename(test.filename)} and {secure_filename(actual_labels.filename)} doesn't have same no. of rows", 'danger')
                return redirect(url_for('home'))
            if df_labels.shape[0] == 0:
                logging.error(f"user#{rand_id}: {secure_filename(actual_labels.filename)} has no rows")
                flash(f'{secure_filename(actual_labels.filename)} has no rows', 'danger')
                return redirect(url_for('home'))
            if df_labels.shape[1] > 1:
                logging.error(f"user#{rand_id}: {secure_filename(actual_labels.filename)} has more than one column")
                flash(f'{secure_filename(actual_labels.filename)} has more than one column', 'danger')
                return redirect(url_for('home'))
            if df_labels.shape[1] == 0:
                logging.error(f"user#{rand_id}: No columns found in {secure_filename(actual_labels.filename)}")
                flash(f'No columns found in {secure_filename(actual_labels.filename)}', 'danger')
                return redirect(url_for('home'))
            if len(df_labels.columns.values.tolist()) == 0:
                logging.error(f"user#{rand_id}: No columns found in {secure_filename(actual_labels.filename)}")
                flash(f'No columns found in {secure_filename(actual_labels.filename)}', 'danger')
                return redirect(url_for('home'))

        df_test['Text'] = df_test['Text'].astype(str)
        logging.info(f'user#{rand_id}: test file and actual label file checked successfully')
        logging.info(f'user#{rand_id}: No. of rows of test dataset uploaded: {df_test.shape[0]}')
        logging.info(f'user#{rand_id}: No. of columns of test dataset uploaded: {df_test.shape[1]}')
        logging.info(f"user#{rand_id}: Datatype of the column of the test dataset uploaded: {df_test['Text'].dtype}")
        multiple_upload = MultipleUpload(df_test)
        tokenized_multi_text = multiple_upload.process()
        predictions_multi = multiple_upload.predict(tokenized_multi_text, m_model)
        encoded_pred = [np.argmax(pred) for pred in predictions_multi]
        label_multi = label_encoder.inverse_transform(encoded_pred)
        info_dict['data'] = df_test['Text'].tolist()
        info_dict['pred'] = label_multi.tolist()
        info_dict['labels'] = list(label_encoder.classes_)
        if secure_filename(actual_labels.filename) != '':
            df_labels['Category'] = df_labels['Category'].astype(str)
            logging.info(f'user#{rand_id}: No. of rows of label dataset uploaded: {df_labels.shape[0]}')
            logging.info(f'user#{rand_id}: No. of columns of label dataset uploaded: {df_labels.shape[1]}')
            logging.info(f"user#{rand_id}: Datatype of the column of the label dataset uploaded: {df_test['Category'].dtype}")
            metrics_multi = multiple_upload.metrics(label_multi.tolist(), df_labels['Category'].tolist())
            # print(metrics_multi['classification_report'])
            info_dict['metrics'] = metrics_multi
            info_dict['actual'] = df_labels['Category'].tolist()
        encrypted_info_dict = json.dumps(info_dict).encode('utf-8')
        flash(f'{secure_filename(test.filename)} and {secure_filename(actual_labels.filename)} uploaded successfully', 'success')
        logging.info(f'user#{rand_id}: Multiple Form Encoded Data Dictionary Sent')
        return redirect(url_for('stat_metrics', data_dict=b64encode(encrypted_info_dict)))

    elif demo_form.validate_on_submit():
        logging.info(f'user#{rand_id}: Demo Form Validated Successfully')
        d_model = demo_form.d_model.data
        info_dict = {}

        # Accessing data from Astra DB database
        try:
            respond_test = astra_client.request(
                method=http_methods.GET,
                path=f"/api/rest/v2/keyspaces/{ASTRA_DB_KEYSPACE}/{ASTRA_DB_COLLECTION_1}/rows")
            db_test = [item['Text'] for item in respond_test['data']]
            logging.info(f'user#{rand_id}: Accessed Text values from database successfully')
            # print(respond_test)
            # print(db_test)
        except Exception:
            logging.error(f'user#{rand_id}: Database Text value access unsuccessfully')
            flash(f'Text Data Access from Database Unsuccessful.', 'danger')
            return redirect(url_for('home'))

        try:
            respond_labels = astra_client.request(
                method=http_methods.GET,
                path=f"/api/rest/v2/keyspaces/{ASTRA_DB_KEYSPACE}/{ASTRA_DB_COLLECTION_2}/rows")
            db_labels = [item['Category'] for item in respond_labels['data']]
            # print(respond_labels)
            # print(db_labels)
        except Exception:
            logging.error(f'user#{rand_id}: Database label value access unsuccessfully')
            flash(f'Label Data Access from Database Unsuccessful.', 'danger')
            return redirect(url_for('home'))

        df_test = pd.DataFrame(columns=['Text'])
        df_labels = pd.DataFrame(columns=['Category'])

        df_test['Text'] = db_test
        df_labels['Category'] = db_labels

        logging.info(f'user#{rand_id}: test and labels read successfully from database.')
        multiple_upload = MultipleUpload(df_test)
        tokenized_multi_text = multiple_upload.process()
        predictions_multi = multiple_upload.predict(tokenized_multi_text, d_model)
        encoded_pred = [np.argmax(pred) for pred in predictions_multi]
        label_multi = label_encoder.inverse_transform(encoded_pred)
        info_dict['data'] = df_test['Text'].tolist()
        info_dict['pred'] = label_multi.tolist()
        info_dict['labels'] = list(label_encoder.classes_)
        metrics_multi = multiple_upload.metrics(label_multi.tolist(), df_labels['Category'].tolist())
        # print(metrics_multi['classification_report'])
        info_dict['metrics'] = metrics_multi
        info_dict['actual'] = df_labels['Category'].tolist()
        encrypted_info_dict = json.dumps(info_dict).encode('utf-8')
        flash('File Upload Successful', 'success')
        logging.info(f'user#{rand_id}: Demo Form Encoded Data Dictionary Sent')
        return redirect(url_for('stat_metrics', data_dict=b64encode(encrypted_info_dict)))

    return render_template('news.html', single_form=single_form,
                           single_multiple_choice_form=single_multiple_choice_form,
                           multiple_form=multiple_form, demo_form=demo_form)


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0")
