import torch
import librosa
import numpy as np
import json
import os
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa class (Giữ nguyên như cũ)
class MusicGenreCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(MusicGenreCNN, self).__init__()
        # Khối 1: Input (1, 128, 300) -> Output (32, 64, 150)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Khối 2: -> Output (64, 32, 75)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Khối 3: -> Output (128, 16, 37)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Khối 4: Thêm khối sâu để học đặc trưng phức tạp -> Output (256, 8, 18)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling: Biến (256, 8, 18) thành (256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Phân loại: Chỉ tốn rất ít tham số so với bản cũ
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1) # Flatten vector 256 chiều
        x = self.classifier(x)
        return x

# 1. CẤU HÌNH
MODEL_PATH = "best_model.pth"
AUDIO_PATH = "./tests/rap_god.mp3"
LABEL_MAP_PATH = "./data/label_map.json"
STATS_PATH = "./data/stats.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. LOAD CONFIG & MODEL
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

with open(STATS_PATH, 'r') as f:
    stats = json.load(f)
means = np.array(stats['mean']).reshape(128, 1)
stds = np.array(stats['std']).reshape(128, 1)

model = MusicGenreCNN(num_classes=8).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 3. HÀM DỰ ĐOÁN 30 GIÂY (CHIA 4 ĐOẠN)
def predict_30s(audio_path, model, inv_label_map, offset=0.0):
    y_full, sr = librosa.load(audio_path, sr=22050, offset=offset, duration=30.0)
    
    segment_duration = 7.0 
    samples_per_segment = int(segment_duration * sr)
    all_outputs = []

    for i in range(4):
        start = i * samples_per_segment
        y_seg = y_full[start : start + samples_per_segment]
        if len(y_seg) < samples_per_segment:
            y_seg = np.pad(y_seg, (0, samples_per_segment - len(y_seg)))

        mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        if mel_db.shape[1] > 300: mel_db = mel_db[:, :300]
        else: mel_db = np.pad(mel_db, ((0,0), (0, 300 - mel_db.shape[1])))

        mel_norm = (mel_db - means) / stds
        input_tensor = torch.tensor(mel_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            all_outputs.append(probs)

    # 1. Tính trung bình xác suất của 4 đoạn
    final_probs = torch.mean(torch.stack(all_outputs), dim=0) # Shape: (1, 8)
    
    # 2. Lấy Top 3 kết quả cao nhất
    top_probs, top_indices = torch.topk(final_probs, k=3, dim=1)
    
    # 3. Chuyển kết quả sang dạng List các Tuple (Genre, Percentage)
    results = []
    for i in range(3):
        genre = inv_label_map[top_indices[0][i].item()]
        prob = top_probs[0][i].item() * 100
        results.append((genre, prob))
        
    return results

# --- PHẦN GỌI HÀM VÀ IN KẾT QUẢ ---
print(f"--- Đang phân tích file: {os.path.basename(AUDIO_PATH)} ---")
try:
    top3_results = predict_30s(AUDIO_PATH, model, inv_label_map, offset=268.0)
    
    print("\nKẾT QUẢ DỰ ĐOÁN (TOP 3):")
    for genre, prob in top3_results:
        print(f"{genre.upper()} - {prob:.2f}%")

except Exception as e:
    print(f"Lỗi: {e}")