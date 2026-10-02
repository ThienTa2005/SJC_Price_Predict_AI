"""Inference uses exactly the architecture and saved scalers from the notebook."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

class VanillaRNN(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, num_layers=1, nonlinearity="tanh", batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        sequence, _ = self.rnn(x)
        return self.head(self.dropout(sequence[:, -1, :]))

class LSTMModel(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        sequence, _ = self.rnn(x)
        return self.head(self.dropout(sequence[:, -1, :]))


def model_directory(variant="price"):
    if variant not in ("price", "log_return", "lstm"):
        raise ValueError("Cách dự đoán không hợp lệ.")
    folder = ROOT / {"price": "model", "log_return": "model/log_return", "lstm": "model/lstm"}[variant]
    required = ["metadata.json", "pytorch_best.pt", "keras_best.keras",
                "test_predictions.csv", "model_comparison.csv"]
    if not all((folder / name).is_file() for name in required):
        raise FileNotFoundError("Chưa có đủ model " + ("LSTM" if variant == "lstm" else variant) + " đã huấn luyện. Hãy chạy notebook tương ứng và chép checkpoint vào website.")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    transform = metadata.get("target_transform", "price")
    if transform != ("price" if variant == "lstm" else variant):
        raise ValueError("Checkpoint và cách xử lý dữ liệu không khớp.")
    if metadata.get("architecture", "rnn") != ("lstm" if variant == "lstm" else "rnn"):
        raise ValueError("Kiến trúc checkpoint không khớp lựa chọn.")
    if variant == "log_return":
        if metadata.get("input_transform") != "log_return" or metadata.get("sequence_length") != metadata["lookback"] - 1:
            raise ValueError("Cấu hình cửa sổ log return không hợp lệ.")
    return folder


def load_bundle(framework="PyTorch", variant="price"):
    if framework not in ("PyTorch", "Keras"):
        raise ValueError("Model không hợp lệ.")
    folder = model_directory(variant)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if framework == "Keras":
        os.environ["KERAS_BACKEND"] = "torch"
        import keras
        model = keras.models.load_model(folder / "keras_best.keras", compile=False)
        return {"model": model, "metadata": metadata, "framework": framework}
    cfg = metadata["config"]
    torch.set_num_threads(1)
    model_class = LSTMModel if variant == "lstm" else VanillaRNN
    model = model_class(cfg["hidden"], cfg["dropout"])
    model.load_state_dict(torch.load(folder / "pytorch_best.pt", map_location="cpu", weights_only=True))
    model.eval()
    return {"model": model, "metadata": metadata, "framework": framework}

def prepare_data(frame, metadata, group):
    if group not in metadata["scalers"]:
        raise ValueError("Nhóm này chưa được model hỗ trợ.")
    target = metadata["target"]
    if "Date" not in frame.columns and "timestamp" in frame.columns:
        frame = frame.rename(columns={"timestamp": "Date"})
    required = ["Date", target] + (["region", "type"] if target == "AveragePrice" else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("CSV thiếu cột: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    if target == "AveragePrice":
        region, kind = json.loads(group)
        frame = frame.loc[(frame.region == region) & (frame.type == kind)].copy()
    if len(frame) < metadata["lookback"]:
        raise ValueError(f"Cần ít nhất {metadata['lookback']} quan sát cho chuỗi đã chọn.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if frame.Date.isna().any():
        raise ValueError("CSV chứa ngày không hợp lệ.")
    if frame.Date.duplicated().any():
        raise ValueError("CSV có ngày trùng trong chuỗi đã chọn.")
    if frame[target].map(lambda x: isinstance(x, (bool, np.bool_))).any():
        raise ValueError("Giá phải là số, không phải True/False.")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    values = frame[target].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Giá phải là số không âm và hữu hạn.")
    if metadata.get("target_transform") == "log_return" and (values <= 0).any():
        raise ValueError("Model log return yêu cầu mọi giá lớn hơn 0.")
    return frame.sort_values("Date").reset_index(drop=True)

def predict(prices, bundle, group):
    metadata = bundle["metadata"]
    if group not in metadata["scalers"]:
        raise ValueError("Nhóm này chưa được model hỗ trợ.")
    if not isinstance(prices, (list, tuple, np.ndarray)) or any(isinstance(x, (bool, np.bool_)) for x in prices):
        raise ValueError("Cần một danh sách giá dạng số.")
    try:
        values = np.asarray(prices, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Giá phải là số.") from exc
    if values.shape != (metadata["lookback"],) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Cần đúng {metadata['lookback']} giá không âm và hữu hạn.")
    scaler = metadata["scalers"][group]
    log_return = metadata.get("target_transform") == "log_return"
    if log_return and (values <= 0).any():
        raise ValueError("Model log return yêu cầu mọi giá lớn hơn 0.")
    features = np.diff(np.log(values)) if log_return else values
    z = ((features - scaler["mean"]) / scaler["scale"]).astype(np.float32).reshape(1, -1, 1)
    with torch.inference_mode():
        if bundle.get("framework") == "Keras":
            output = bundle["model"](z, training=False)
            output = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
            normalized = float(output.reshape(-1)[0])
        else:
            normalized = float(bundle["model"](torch.from_numpy(z)).item())
        result = normalized * scaler["scale"] + scaler["mean"]
    if log_return:
        with np.errstate(over="ignore", invalid="ignore"):
            result = values[-1] * np.exp(result)
        if result <= 0:
            raise ValueError("Model log return trả về giá không hợp lệ.")
    if not np.isfinite(result):
        raise ValueError("Model trả về giá trị không hữu hạn.")
    return result



def evaluation_metrics(evaluation, tolerance_percent=5.0, architecture="RNN"):
    """Evaluate both models on the same saved holdout rows; exclude zero actuals from relative errors."""
    actual = evaluation["actual"].to_numpy(dtype=float)
    rows = []
    for model in (f"PyTorch {architecture}", f"Keras {architecture}"):
        predicted = evaluation[model].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(predicted)
        a, p = actual[valid], predicted[valid]
        error = np.abs(p - a)
        relative = error[a != 0] / np.abs(a[a != 0]) * 100
        rows.append({"Model": model, "Đúng trong ngưỡng (%)": float(np.mean(relative <= tolerance_percent) * 100) if len(relative) else np.nan,
                     "MAE (triệu VND/lượng)": float(np.mean(error)) if len(error) else np.nan,
                     "RMSE (triệu VND/lượng)": float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan,
                     "MAPE (%)": float(np.mean(relative)) if len(relative) else np.nan,
                     "Số mẫu": len(a), "Mẫu tính tỉ lệ": len(relative)})
    return pd.DataFrame(rows)
