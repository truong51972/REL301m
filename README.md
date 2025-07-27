# Spotify Playlist Generation with Reinforcement Learning

## Dự án tạo Playlist thông minh sử dụng Reinforcement Learning

Dự án này sử dụng Deep Q-Network (DQN) để tạo ra các playlist âm nhạc đa dạng và thú vị từ dữ liệu Spotify.

## Cấu trúc Project

```
Final_REL/
├── src/                    # Source code chính
├── web/                   # Web application  
├── experiments/          # Các thí nghiệm training
├── data/                 # Dữ liệu
├── models/               # Các model đã train
├── logs/                 # Log files
└── docs/                 # Documentation
```

## Quick Start

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu
```bash
cd src
python collect_data.py
```

### 3. Training model
```bash
cd experiments
python train_diversity_focused.py
```

### 4. Chạy web app
```bash
cd web
python app.py
```

## Tính năng chính

- **Reinforcement Learning**: Sử dụng DQN để tạo playlist
- **Diversity Focused**: Tối ưu hóa đa dạng âm nhạc
- **Web Interface**: Giao diện web để tạo playlist
- **Spotify Integration**: Kết nối với Spotify API

## Model Performance

- **Diversity Score**: >2.0 (target)
- **Similarity**: <0.9 (adjusted)
- **Training Time**: ~2.6 minutes cho 30 episodes
- **Memory Buffer**: 570 experiences

## Documentation

Xem thêm chi tiết trong folder `docs/`:
- **[Hướng dẫn cho Giảng viên](docs/INSTRUCTOR_GUIDE.md)** - **ĐỌC TRƯỚC**
- [Hướng dẫn nhanh](docs/QUICK_START.md)
- [Ghi chú chi tiết về Training](docs/TRAINING_EXPERIMENTS_NOTES.md)
- [Tóm tắt Training](docs/TRAINING_SUMMARY.md)
- **[Báo cáo Hardcode Parameters](docs/HARDCODED_PARAMETERS_REPORT.md)** - **QUAN TRỌNG**
- **[So sánh số liệu 3 phiên bản](docs/TRAINING_COMPARISON.md)** - **RẤT QUAN TRỌNG**

## Kết quả

Model đã được train thành công với:
- 30 episodes training
- Diversity score đạt mục tiêu >2.0
- Similarity được kiểm soát dưới 0.9
- Web interface hoạt động ổn định

---

**Tác giả**: Dự án Reinforcement Learning cho Spotify Playlist Generation 