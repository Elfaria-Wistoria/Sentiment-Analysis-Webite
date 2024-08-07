from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
import joblib
import re
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from googleapiclient.errors import HttpError
import os

# Inisialisasi Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Menambahkan kunci rahasia untuk sesi

# Konfigurasi Flask-Session untuk menyimpan sesi di file sistem
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'  # Pastikan direktori ini ada
Session(app)

# Load model dan tokenizer
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
transformer_model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
loaded_model = joblib.load('svm_model.pkl')
loaded_label_encoder = joblib.load('label_encoder.pkl')

# Stopwords untuk bahasa Inggris
stop_words = set(stopwords.words('english'))

# Fungsi untuk pembersihan teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Hapus tanda baca dan angka
    text = re.sub(r'\s\s+', ' ', text)  # Hapus spasi berlebih
    text = text.strip()  # Hapus spasi di awal dan akhir
    return text

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

def preprocess_and_get_embedding(new_comment):
    cleaned_text = clean_text(new_comment.lower())
    tokenized_text = word_tokenize(cleaned_text)
    filtered_text = ' '.join(remove_stopwords(tokenized_text))
    
    inputs = tokenizer(filtered_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = transformer_model(**inputs)
    embedding = outputs.logits.detach().numpy().flatten()
    
    return embedding

# Fungsi untuk memprediksi sentimen menggunakan model transformer
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = transformer_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_index = torch.argmax(probs, dim=-1).item()
    sentiment_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
    return sentiment_map[sentiment_index]

# Fungsi untuk mencari video berdasarkan kata kunci
def search_videos(query, max_results):
    api_key = 'AIzaSyAAPg5vLDAON71Wqpl7x-xslcNVI5o_IRA'
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        part='snippet',
        q=query,
        type='video',
        maxResults=max_results
    )
    response = request.execute()
    video_ids = [item['id']['videoId'] for item in response['items']]
    print(f"Video IDs found: {video_ids}")  # Logging untuk video ID
    return video_ids

# Fungsi untuk mendapatkan komentar dari video
def get_comments_from_videos(video_ids, num_comments):
    api_key = 'AIzaSyAAPg5vLDAON71Wqpl7x-xslcNVI5o_IRA'
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []

    for video_id in video_ids:
        print(f"Processing video ID: {video_id}")  # Logging untuk video ID yang sedang diproses
        try:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=num_comments
            )
            response = request.execute()
            fetched_comments = len(response.get('items', []))
            print(f"Number of comments fetched: {fetched_comments}")  # Logging jumlah komentar yang diambil

            for item in response['items']:
                comment_data = item['snippet']['topLevelComment']['snippet']
                comment = {
                    'comment': comment_data['textDisplay'],
                    'author': comment_data['authorDisplayName'],
                    'published_at': comment_data['publishedAt'],
                    'like_count': comment_data['likeCount'],
                    'video_id': video_id
                }
                comments.append(comment)
        except HttpError as e:
            error_message = e._get_reason()
            print(f"Error for video {video_id}: {error_message}")
            continue

    return comments

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        max_videos = int(request.form['max_videos'])
        num_comments = int(request.form['num_comments'])
        
        # Mencari video dan mendapatkan komentar
        video_ids = search_videos(query, max_videos)
        comments = get_comments_from_videos(video_ids, num_comments)
        
        # Debug: Print total comments fetched
        print(f"Total comments fetched: {len(comments)}")

        # Simpan komentar dalam sesi
        session['comments'] = comments
        return redirect(url_for('process'))

    return render_template('index.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    comments = session.get('comments', [])

    if request.method == 'POST':
        # Proses komentar dan simpan embedding serta label sentimen
        processed_comments = []
        for comment in comments:
            embedding = preprocess_and_get_embedding(comment['comment'])
            sentiment = predict_sentiment(comment['comment'])
            comment['embedding'] = embedding.tolist()
            comment['label'] = sentiment
            processed_comments.append(comment)
        
        # Debug: Check processed comments and their labels
        print(f"Processed {len(processed_comments)} comments with labels")

        # Simpan hasil dalam sesi
        session['processed_comments'] = processed_comments
        return redirect(url_for('model'))

    return render_template('process.html', comments=comments)

@app.route('/model', methods=['GET', 'POST'])
def model():
    processed_comments = session.get('processed_comments', [])
    
    if request.method == 'POST':
        embeddings = np.array([c['embedding'] for c in processed_comments])
        labels = [c['label'] for c in processed_comments]

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)

        # Debug: Print label distribution
        unique, counts = np.unique(y_encoded, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")

        # Melatih model SVM dengan data baru
        model_svm = SVC(kernel='linear', C=1, random_state=42)
        model_svm.fit(embeddings, y_encoded)

        # Prediksi untuk menghitung akurasi
        predictions = model_svm.predict(embeddings)
        accuracy = accuracy_score(y_encoded, predictions)
        session['accuracy'] = accuracy  # Simpan akurasi dalam sesi

        # Simpan model dan encoder ke disk
        joblib.dump(model_svm, 'temp_svm_model.pkl')
        joblib.dump(label_encoder, 'temp_label_encoder.pkl')

        return redirect(url_for('classify'))

    return render_template('model.html', comments=processed_comments)

@app.route('/classify', methods=['GET'])
def classify():
    processed_comments = session.get('processed_comments', [])
    model_svm = joblib.load('temp_svm_model.pkl')
    label_encoder = joblib.load('temp_label_encoder.pkl')
    accuracy = session.get('accuracy', 0)  # Dapatkan akurasi dari sesi

    results = []
    for comment in processed_comments:
        embedding = np.array(comment['embedding']).reshape(1, -1)
        prediction = model_svm.predict(embedding)
        sentiment = label_encoder.inverse_transform(prediction)[0]
        comment['sentiment'] = sentiment
        results.append(comment)

    return render_template('classify.html', results=results, accuracy=accuracy)

@app.route('/test', methods=['GET', 'POST'])
def test():
    sentiment = None
    if request.method == 'POST':
        test_comment = request.form['test_comment']
        # Preprocess the comment
        embedding = preprocess_and_get_embedding(test_comment).reshape(1, -1)
        model_svm = joblib.load('temp_svm_model.pkl')
        label_encoder = joblib.load('temp_label_encoder.pkl')
        # Predict sentiment
        prediction = model_svm.predict(embedding)
        sentiment = label_encoder.inverse_transform(prediction)[0]

    return render_template('test.html', sentiment=sentiment)

if __name__ == '__main__':
    # Bersihkan file model sementara jika ada saat memulai ulang aplikasi
    if os.path.exists('temp_svm_model.pkl'):
        os.remove('temp_svm_model.pkl')
    if os.path.exists('temp_label_encoder.pkl'):
        os.remove('temp_label_encoder.pkl')
    
    app.run(debug=True)
