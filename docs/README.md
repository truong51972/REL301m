# Spotify Playlist Generation with Reinforcement Learning

## Cấu trúc Project

```
Final_REL/
├── src/                    # Source code chính
│   ├── models.py          # Định nghĩa các model DQN
│   ├── config.py          # Cấu hình hệ thống
│   └── collect_data.py    # Thu thập dữ liệu từ Spotify
├── web/                   # Web application
│   ├── app.py            # Flask web app
│   ├── run.py            # Server runner
│   ├── playlist_generator.py  # Generator cho web
│   └── templates/        # HTML templates
│       └── index.html
├── experiments/          # Các thí nghiệm training
│   ├── train_simple.py
│   ├── train_improved.py
│   ├── train_diversity_focused.py
│   ├── train_detailed.py
│   ├── train_model.py
│   ├── train_model_ascii.py
│   ├── train_model_safe.py
│   ├── train_improved_fixed.py
│   ├── simple_model_check.py
│   └── plot_loss.py
├── data/                 # Dữ liệu
│   ├── spotify_songs.json
│   └── embeddings.json
├── models/               # Các model đã train
│   ├── dqn_diversity_model.h5
│   ├── dqn_improved_model.h5
│   ├── improved_fixed_dqn_model.h5
│   └── improved_dqn_model.h5
├── logs/                 # Log files
│   ├── loss_log.txt
│   ├── log_train.txt
│   ├── log_train_detailed.txt
│   ├── log_train_improved.txt
│   ├── log_train_improved_fixed.txt
│   └── log_train_diversity_focused.txt
├── docs/                 # Documentation
│   ├── README.md
│   ├── QUICK_START.md
│   ├── BAI_THUYET_TRINH.md
│   ├── BAO_CAO_DU_AN.md
│   ├── SLIDE_GUIDE.md
│   └── SLIDE_CODE_IMPLEMENTATION.md
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore rules
```

## Cách sử dụng

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

## Tác giả

Dự án Reinforcement Learning cho Spotify Playlist Generation 