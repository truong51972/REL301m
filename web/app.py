from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import os
from datetime import datetime
from playlist_generator import PlaylistGenerator
from spotify_data_collector import SpotifyDataCollector
from models import DQNModel, PlaylistEnvironment
from config import Config
import threading
import time
import subprocess
import sys

app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET_KEY
CORS(app)

# Global variables
generator = None
data_collector = None
is_training = False
training_progress = 0
is_collecting = False
collecting_progress = 0
collecting_message = ""
training_logs = []  # Lưu log training

def add_training_log(message, log_type="INFO"):
    """Thêm log training với timestamp và phân loại"""
    global training_logs
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    training_logs.append(log_entry)
    
    # Giới hạn số lượng log để tránh quá tải
    if len(training_logs) > 1000:
        training_logs = training_logs[-500:]  # Giữ lại 500 log gần nhất
    
    print(log_entry, flush=True)

def initialize_generator():
    """Khởi tạo generator nếu có dữ liệu sẵn"""
    global generator
    try:
        if os.path.exists(Config.PLAYLIST_DATA_FILE):
            generator = PlaylistGenerator()
            if generator.load_data():
                print(f"Đã load {len(generator.songs_data)} bài hát từ file")
                
                # Tự động load model nếu có
                model_path = 'models/dqn_model.h5'
                if os.path.exists(model_path):
                    try:
                        generator.load_model(model_path)
                        print("✅ Đã load model thành công!")
                        return True
                    except Exception as e:
                        print(f"⚠️ Không thể load model: {e}")
                        return True  # Vẫn OK nếu có dữ liệu
                else:
                    print("⚠️ Chưa có model, cần training trước")
                    return True  # Vẫn OK nếu có dữ liệu
    except Exception as e:
        print(f"Lỗi khi load dữ liệu: {e}")
    return False

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Lấy trạng thái hệ thống"""
    global generator, is_training, training_progress, is_collecting, collecting_progress, collecting_message
    
    # Kiểm tra xem dữ liệu có thực sự tồn tại không
    data_exists = False
    songs_count = 0
    
    if generator and generator.songs_data:
        # Sử dụng số lượng bài hát thực tế từ generator
        songs_count = len(generator.songs_data)
        data_exists = songs_count > 0
    else:
        # Kiểm tra file dữ liệu có tồn tại không
        if os.path.exists(Config.PLAYLIST_DATA_FILE):
            try:
                with open(Config.PLAYLIST_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    songs_count = len(data)
                    data_exists = songs_count > 0
            except:
                data_exists = False
                songs_count = 0
        else:
            data_exists = False
            songs_count = 0
    
    # Reset generator nếu dữ liệu không tồn tại
    if not data_exists and generator:
        generator = None
    
    status = {
        'data_loaded': data_exists,
        'model_trained': generator is not None and generator.dqn_model is not None,
        'is_training': is_training,
        'training_progress': training_progress,
        'is_collecting': is_collecting,
        'collecting_progress': collecting_progress,
        'collecting_message': collecting_message,
        'songs_count': songs_count
    }
    
    return jsonify(status)

@app.route('/api/collect-data', methods=['POST'])
def collect_data():
    """Thu thập dữ liệu từ Spotify - Chỉ thông báo"""
    return jsonify({
        'success': False, 
        'message': 'Vui lòng chạy collect_data.py để thu thập dữ liệu trước khi sử dụng web interface'
    })

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train model - Chỉ thông báo"""
    return jsonify({
        'success': False, 
        'message': 'Vui lòng chạy train_model.py để training model trước khi sử dụng web interface'
    })

@app.route('/api/generate-playlist', methods=['POST'])
def generate_playlist():
    """Tạo playlist"""
    global generator
    
    try:
        if generator is None or generator.dqn_model is None:
            return jsonify({'success': False, 'message': 'Vui lòng train model trước'})
        
        data = request.json
        length = data.get('length', 20)
        seed_song = data.get('seed_song', None)
        constraints = data.get('constraints', {})
        
        playlist = generator.generate_playlist(
            seed_song_id=seed_song,
            length=length,
            constraints=constraints
        )
        
        # Đánh giá playlist
        score = generator.evaluate_playlist(playlist)
        
        # Format playlist cho frontend
        formatted_playlist = []
        for song in playlist:
            formatted_song = {
                'id': song['id'],
                'name': song['name'],
                'artist': song['artist'],
                'album': song['album'],
                'genre': song.get('genre', 'Unknown'),
                'popularity': song.get('popularity', 0),
                'release_date': song.get('release_date', 'Unknown')
            }
            formatted_playlist.append(formatted_song)
        
        return jsonify({
            'success': True,
            'playlist': formatted_playlist,
            'score': score,
            'length': len(formatted_playlist)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})

@app.route('/api/training-logs')
def get_training_logs():
    """Lấy log training"""
    global training_logs
    return jsonify({'logs': training_logs})

@app.route('/api/clear-training-logs', methods=['POST'])
def clear_training_logs():
    """Xóa log training"""
    global training_logs
    training_logs = []
    return jsonify({'success': True, 'message': 'Đã xóa log training'})

@app.route('/api/search-songs', methods=['GET'])
def search_songs():
    """Tìm kiếm bài hát"""
    global generator
    
    if generator is None:
        return jsonify({'success': False, 'message': 'Chưa có dữ liệu'})
    
    query = request.args.get('q', '').lower()
    limit = int(request.args.get('limit', 10))
    
    results = []
    for song in generator.songs_data:
        if (query in song['name'].lower() or 
            query in song['artist'].lower() or
            query in song.get('genre', '').lower()):
            results.append({
                'id': song['id'],
                'name': song['name'],
                'artist': song['artist'],
                'genre': song.get('genre', 'Unknown')
            })
            if len(results) >= limit:
                break
    
    return jsonify({'success': True, 'songs': results})

@app.route('/api/genres')
def get_genres():
    """Lấy danh sách thể loại từ dữ liệu thực tế + các thể loại cổ điển"""
    global generator
    
    # Các thể loại cổ điển/phổ biến
    standard_genres = ['Vietnamese', 'Pop', 'Rock', 'Hip-hop', 'Electronic', 'Jazz', 'Classical', 'Country', 'R&B', 'Folk', 'Reggae', 'Blues', 'Metal']
    
    if generator and generator.songs_data:
        # Lấy các genre thực tế từ dữ liệu
        data_genres = set()
        for song in generator.songs_data:
            genre = song.get('genre', 'Unknown')
            if genre and genre != 'Unknown':
                data_genres.add(genre)
        
        # Kết hợp genres từ dữ liệu với standard genres
        all_genres = set(standard_genres)
        all_genres.update(data_genres)
        
        # Chuyển thành list, đặt Vietnamese lên đầu
        genres_list = list(all_genres)
        if 'Vietnamese' in genres_list:
            genres_list.remove('Vietnamese')
        genres_list = ['Vietnamese'] + sorted(genres_list)
            
        return jsonify({'genres': genres_list})
    else:
        # Fallback nếu chưa có dữ liệu
        return jsonify({'genres': standard_genres})

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset hoàn toàn hệ thống"""
    global generator, data_collector, is_training, training_progress, is_collecting, collecting_progress, collecting_message
    
    try:
        # Reset các biến global
        generator = None
        data_collector = None
        is_training = False
        training_progress = 0
        is_collecting = False
        collecting_progress = 0
        collecting_message = ""
        
        # Xóa các file dữ liệu
        files_to_delete = [
            Config.PLAYLIST_DATA_FILE,
            Config.EMBEDDING_FILE,
            Config.RAW_DATA_FILE,
            Config.METRICS_FILE
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Xóa model files
        model_files = ['models/dqn_model.h5']
        for model_file in model_files:
            if os.path.exists(model_file):
                os.remove(model_file)
        
        return jsonify({'success': True, 'message': 'Đã reset hệ thống thành công'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Lỗi khi reset: {str(e)}'})



if __name__ == '__main__':
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    # Khởi tạo generator nếu có dữ liệu sẵn
    print("Khởi tạo hệ thống...")
    initialize_generator()
    
    app.run(debug=False, host='0.0.0.0', port=5000) 