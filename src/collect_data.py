#!/usr/bin/env python3
"""
Script thu thập dữ liệu từ Spotify API
Chạy độc lập, không cần web interface
"""

import os
import json
import time
from spotify_data_collector import SpotifyDataCollector
from config import Config

def main():
    print("🎵 KIỂM TRA VÀ THU THẬP DỮ LIỆU TỪ SPOTIFY")
    print("=" * 50)
    
    # Kiểm tra xem đã có dữ liệu chưa
    if os.path.exists(Config.PLAYLIST_DATA_FILE):
        try:
            with open(Config.PLAYLIST_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Đã tìm thấy dữ liệu hiện có: {len(data)} bài hát")
                
                # Thống kê dữ liệu hiện có
                genres = {}
                for song in data:
                    genre = song.get('genre', 'Unknown')
                    genres[genre] = genres.get(genre, 0) + 1
                
                print("📈 Thống kê dữ liệu hiện có:")
                for genre, count in sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"   - {genre}: {count} bài hát")
                
                choice = input("\n❓ Bạn có muốn thu thập lại dữ liệu không? (y/N): ").lower().strip()
                if choice != 'y':
                    print("✅ Giữ nguyên dữ liệu hiện có!")
                    return
                else:
                    print("🔄 Bắt đầu thu thập lại dữ liệu...")
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc dữ liệu hiện có: {e}")
            print("🔄 Bắt đầu thu thập dữ liệu mới...")
    else:
        print("📊 Chưa có dữ liệu, bắt đầu thu thập...")
    
    # Khởi tạo collector
    collector = SpotifyDataCollector()
    
    # Thu thập dữ liệu
    print("\n📊 Đang thu thập dữ liệu bài hát...")
    print("   - Nhạc Việt Nam")
    print("   - Nhạc quốc tế")
    print("   - Các thể loại khác nhau")
    print()
    
    def progress_callback(current, total, message):
        progress = (current / total) * 100
        print(f"   [{current}/{total}] {progress:.1f}% - {message}")
    
    try:
        # Thu thập dữ liệu (2000 bài hát)
        songs = collector.collect_data(target_count=2000, progress_callback=progress_callback)
        
        if songs and len(songs) > 0:
            # Lưu dữ liệu
            collector.save_data(Config.PLAYLIST_DATA_FILE)
            
            print()
            print("✅ THU THẬP DỮ LIỆU THÀNH CÔNG!")
            print(f"📁 Dữ liệu được lưu tại: {Config.PLAYLIST_DATA_FILE}")
            
            # Hiển thị thống kê
            if os.path.exists(Config.PLAYLIST_DATA_FILE):
                with open(Config.PLAYLIST_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"📊 Tổng số bài hát: {len(data)}")
                    
                    # Thống kê theo genre
                    genres = {}
                    for song in data:
                        genre = song.get('genre', 'Unknown')
                        genres[genre] = genres.get(genre, 0) + 1
                    
                    print("📈 Thống kê theo thể loại:")
                    for genre, count in sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10]:
                        print(f"   - {genre}: {count} bài hát")
            
        else:
            print()
            print("❌ THU THẬP DỮ LIỆU THẤT BẠI!")
            print("   Vui lòng kiểm tra lại Spotify API credentials")
            
    except Exception as e:
        print()
        print(f"❌ LỖI: {str(e)}")
        print("   Vui lòng kiểm tra lại cấu hình và thử lại")

if __name__ == "__main__":
    main() 