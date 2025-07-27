#!/usr/bin/env python3
"""
Script khởi động đơn giản cho Controllable Music Playlist Generation
Dành cho người mới bắt đầu
"""

import os
import sys
import subprocess
import time

def print_banner():
    """In banner chào mừng"""
    print("=" * 60)
    print("🎵 Controllable Music Playlist Generation 🎵")
    print("   Hệ thống tạo playlist thông minh với RL")
    print("=" * 60)
    print()

def check_python_version():
    """Kiểm tra phiên bản Python"""
    if sys.version_info < (3, 8):
        print("❌ Lỗi: Cần Python 3.8 trở lên")
        print(f"   Phiên bản hiện tại: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def check_dependencies():
    """Kiểm tra thư viện cần thiết"""
    print("🔍 Kiểm tra thư viện...")
    
    required_packages = [
        'flask', 'spotipy', 'tensorflow', 'numpy', 
        'pandas', 'scikit-learn', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Thiếu")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Thiếu {len(missing_packages)} thư viện")
        install = input("Bạn có muốn cài đặt tự động không? (y/n): ")
        if install.lower() == 'y':
            install_dependencies()
        else:
            print("Vui lòng chạy: pip install -r requirements.txt")
            return False
    else:
        print("✅ Tất cả thư viện đã sẵn sàng")
    
    return True

def install_dependencies():
    """Cài đặt thư viện"""
    print("📦 Đang cài đặt thư viện...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Cài đặt thành công")
        return True
    except subprocess.CalledProcessError:
        print("❌ Lỗi khi cài đặt thư viện")
        return False

def create_directories():
    """Tạo thư mục cần thiết"""
    directories = ['data', 'models', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Tạo thư mục: {directory}")

def show_menu():
    """Hiển thị menu chính"""
    print("\n🎯 Chọn hành động:")
    print("1. 🚀 Khởi động Web UI (Khuyến nghị)")
    print("2. 📊 Thu thập dữ liệu từ Spotify")
    print("3. 🧠 Training model")
    print("4. 🎵 Tạo playlist (Command Line)")
    print("5. 📖 Xem hướng dẫn chi tiết")
    print("6. ❌ Thoát")
    print()

def run_web_ui():
    """Khởi động web UI"""
    print("🌐 Khởi động Web UI...")
    print("   Mở trình duyệt và truy cập: http://localhost:5000")
    print("   Nhấn Ctrl+C để dừng server")
    print()
    
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Đã dừng server")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def collect_data():
    """Thu thập dữ liệu"""
    print("📊 Bắt đầu thu thập dữ liệu từ Spotify...")
    print("   Quá trình này có thể mất 10-15 phút")
    print("   Vui lòng chờ...")
    print()
    
    try:
        from spotify_data_collector import SpotifyDataCollector
        collector = SpotifyDataCollector()
        collector.collect_all_data()
        print("✅ Thu thập dữ liệu thành công!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def train_model():
    """Training model"""
    print("🧠 Bắt đầu training model...")
    print("   Quá trình này có thể mất 30-60 phút")
    print("   Vui lòng chờ...")
    print()
    
    try:
        from playlist_generator import PlaylistGenerator
        generator = PlaylistGenerator()
        if generator.load_data():
            generator.train_rl_model(episodes=500)
            print("✅ Training thành công!")
        else:
            print("❌ Vui lòng thu thập dữ liệu trước")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def generate_playlist_cli():
    """Tạo playlist qua command line"""
    print("🎵 Tạo playlist qua Command Line")
    print()
    
    try:
        from playlist_generator import PlaylistGenerator
        generator = PlaylistGenerator()
        
        if not generator.load_data():
            print("❌ Vui lòng thu thập dữ liệu trước")
            return
        
        if not generator.dqn_model:
            print("❌ Vui lòng training model trước")
            return
        
        # Nhập thông tin
        length = int(input("Độ dài playlist (5-50): "))
        genre = input("Thể loại (pop/rock/vietnamese/...): ").lower()
        min_popularity = int(input("Độ phổ biến tối thiểu (0-100): "))
        
        constraints = {
            'genre': [genre] if genre else [],
            'min_popularity': min_popularity
        }
        
        print("\n🎵 Đang tạo playlist...")
        playlist = generator.generate_playlist(length=length, constraints=constraints)
        
        print(f"\n✅ Tạo thành công {len(playlist)} bài hát:")
        for i, song in enumerate(playlist, 1):
            print(f"{i:2d}. {song['name']} - {song['artist']} ({song.get('genre', 'Unknown')})")
        
        score = generator.evaluate_playlist(playlist)
        print(f"\n📊 Điểm đánh giá: {score:.2f}/10")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def show_guide():
    """Hiển thị hướng dẫn"""
    print("\n📖 HƯỚNG DẪN CHI TIẾT")
    print("=" * 40)
    print()
    print("🎯 QUY TRÌNH SỬ DỤNG:")
    print("1. Thu thập dữ liệu từ Spotify (1 lần duy nhất)")
    print("2. Training model (1 lần duy nhất)")
    print("3. Tạo playlist (nhiều lần)")
    print()
    print("⚡ CÁCH NHANH NHẤT:")
    print("1. Chọn 'Khởi động Web UI'")
    print("2. Trên web, nhấn 'Thu thập dữ liệu'")
    print("3. Nhấn 'Bắt đầu Training'")
    print("4. Tạo playlist với giao diện đẹp")
    print()
    print("🔧 CẤU HÌNH:")
    print("- Spotify API đã được cấu hình sẵn")
    print("- Sẽ thu thập ~10,000 bài hát")
    print("- Hỗ trợ nhạc Việt Nam và quốc tế")
    print()
    print("📁 DỮ LIỆU:")
    print("- Lưu trong thư mục 'data/'")
    print("- Model lưu trong thư mục 'models/'")
    print("- Có thể xóa và thu thập lại")
    print()
    input("Nhấn Enter để quay lại menu...")

def main():
    """Hàm main"""
    print_banner()
    
    # Kiểm tra hệ thống
    if not check_python_version():
        return
    
    create_directories()
    
    if not check_dependencies():
        return
    
    print("\n✅ Hệ thống sẵn sàng!")
    
    # Menu chính
    while True:
        show_menu()
        
        try:
            choice = input("Nhập lựa chọn (1-6): ").strip()
            
            if choice == '1':
                run_web_ui()
            elif choice == '2':
                collect_data()
            elif choice == '3':
                train_model()
            elif choice == '4':
                generate_playlist_cli()
            elif choice == '5':
                show_guide()
            elif choice == '6':
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ")
                
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main() 