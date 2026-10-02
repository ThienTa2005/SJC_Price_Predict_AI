# SJC — Website dự đoán giá vàng

Website Streamlit độc lập, cùng cấu trúc với website-amzn và website-avocado.
Dùng RNN PyTorch/Keras đã huấn luyện trong RNN_Gold_PyTorch_Keras.ipynb.
Có dữ liệu mẫu, tải CSV, lịch sử giá, dự đoán bước tiếp theo, so sánh hai model,
ngưỡng sai số và tải kết quả CSV.

## Chạy trên máy

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python start.py
```

Mở http://localhost:8501. Khi chạy cùng website khác, đặt `$env:PORT="8503"` trước khi chạy.

## Dữ liệu và model

CSV cần `timestamp,sell_1l` hoặc `Date,sell_1l`, ít nhất 60 quan sát.
Cột `buy_1l` có thể đi kèm nhưng model chỉ dự đoán giá bán `sell_1l`.
Ngày hợp lệ, không trùng; giá không âm, hữu hạn. Tệp tối đa 10 MB.
Đơn vị triệu VND/lượng được suy ra từ thang giá CSV theo metadata notebook,
không quy đổi hay nhân giá đầu vào. Dữ liệu mẫu là lịch sử, không cập nhật trực tiếp.
Có thể chọn RNN giá trực tiếp hoặc Log return cho cả PyTorch và Keras.
Log return dùng checkpoint từ RNN_Gold_LogReturn_PyTorch_Keras.ipynb, học biến động
logarit giữa các giá liên tiếp rồi quy đổi dự đoán về triệu VND/lượng. CSV vẫn chứa
giá gốc và mọi giá phải lớn hơn 0 khi chọn Log return.
Model và scaler được lấy từ artifacts/gold và artifacts/gold_log_return,
scaler giữ nguyên từ tập huấn luyện.

## Deploy

Upload toàn bộ nội dung folder này vào gốc repo GitHub riêng.
Render Blueprint dùng `render.yaml`; hoặc Web Service với build
`pip install -r requirements.txt`, start `python start.py`, Python 3.11.11,
health check `/_stcore/health`. Có sẵn Dockerfile.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Kiểm tra dự đoán hai framework với kết quả notebook, kiểm tra dữ liệu và thao tác giao diện.
