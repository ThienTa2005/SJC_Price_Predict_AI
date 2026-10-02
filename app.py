from pathlib import Path
import json
import pandas as pd
import streamlit as st
from inference import load_bundle, prepare_data, predict, evaluation_metrics, model_directory

ROOT = Path(__file__).resolve().parent
IS_AVOCADO = json.loads((ROOT / "model/metadata.json").read_text(encoding="utf-8"))["target"] == "AveragePrice"
TITLE = "SJC"
st.set_page_config(page_title=f"{TITLE} · Dự đoán giá", page_icon="🥑" if IS_AVOCADO else "🟡", layout="wide")

@st.cache_resource
def cached_bundle(framework="PyTorch", variant="price"):
    return load_bundle(framework, variant)

@st.cache_data
def read_sample():
    return pd.read_csv(ROOT / "data/prices.csv")

st.caption("PRICE FORECAST · " + TITLE.upper())
st.title(f"Dự đoán giá {TITLE}")
st.write("Chọn dữ liệu, xem lịch sử và dự đoán giá ở bước tiếp theo bằng model đã huấn luyện.")

metadata = json.loads((ROOT / "model/metadata.json").read_text(encoding="utf-8"))
target = metadata["target"]
lookback = metadata["lookback"]

with st.sidebar:
    st.header("Dữ liệu dự đoán")
    selected_model = st.selectbox("Model dự đoán", ["PyTorch", "Keras", "So sánh cả hai"])
    architecture = st.radio("Kiến trúc", ["RNN"], key="architecture")
    methods = ["Giá trực tiếp"] if architecture == "RNN" else ["Giá trực tiếp"]
    method = st.radio("Cách dự đoán", methods, key="prediction_method_" + architecture)
    variant = "lstm" if architecture == "LSTM" else ("log_return" if method == "Log return" else "price")
    try:
        model_dir = model_directory(variant)
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        st.stop()
    target = metadata["target"]
    lookback = metadata["lookback"]
    if variant == "log_return":
        st.caption("Model học biến động tương đối và trả kết quả theo triệu VND/lượng. CSV vẫn chứa giá gốc.")
    source = st.radio("Nguồn dữ liệu", ["Dữ liệu mẫu", "Tải CSV"])
    group = "SJC_Gold"
    if IS_AVOCADO:
        groups = [json.loads(key) for key in metadata["scalers"]]
        regions = sorted({pair[0] for pair in groups})
        region = st.selectbox("Khu vực", regions, index=regions.index("TotalUS"))
        kinds = sorted({pair[1] for pair in groups if pair[0] == region})
        kind = st.selectbox("Loại bơ", kinds)
        group = json.dumps([region, kind], ensure_ascii=False)
    st.caption(f"Model sử dụng {lookback} quan sát gần nhất, theo thứ tự cũ → mới.")
    if source == "Tải CSV":
        upload = st.file_uploader("CSV của bạn (tối đa 10 MB)", type=["csv"])
        st.caption("Cột bắt buộc: " + ("Date, AveragePrice, region, type" if IS_AVOCADO else "timestamp (hoặc Date), sell_1l"))
    else:
        upload = None
    sample = read_sample()
    st.download_button("Tải CSV mẫu", sample.to_csv(index=False).encode("utf-8-sig"), f"{TITLE.lower()}-sample.csv", "text/csv")

if source == "Tải CSV" and upload is None:
    st.info("Tải một CSV ở thanh bên để bắt đầu.")
    st.stop()
try:
    if upload is not None and upload.size > 10 * 1024 * 1024:
        raise ValueError("Tệp vượt quá giới hạn 10 MB.")
    raw = pd.read_csv(upload) if upload is not None else sample
    frame = prepare_data(raw, metadata, group)
except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

window = frame.tail(lookback)
last = float(window[target].iloc[-1])
a, b, c = st.columns(3)
a.metric("Giá quan sát cuối", f"{last:,.4f}")
b.metric("Số quan sát", f"{len(frame):,}")
c.metric("Ngày quan sát cuối", frame.Date.iloc[-1].strftime("%d/%m/%Y"))
st.caption(f"Dữ liệu từ {frame.Date.iloc[0]:%d/%m/%Y} đến {frame.Date.iloc[-1]:%d/%m/%Y}. Dữ liệu mẫu là lịch sử, không cập nhật trực tiếp.")
st.caption("Model dự đoán giá bán sell_1l. Đơn vị triệu VND/lượng được suy ra từ thang giá CSV, theo notebook Gold.")
st.subheader("Lịch sử giá bán vàng SJC")
st.line_chart(frame.set_index("Date")[[target]].rename(columns={target: "Giá (triệu VND/lượng)"}), color="#3b9b76" if IS_AVOCADO else "#e99a26")
with st.expander(f"Xem {lookback} quan sát dùng để dự đoán"):
    st.dataframe(window[["Date", target]], hide_index=True, width="stretch")

frameworks = ["PyTorch", "Keras"] if selected_model == "So sánh cả hai" else [selected_model]
if st.button("Dự đoán bước tiếp theo", type="primary"):
    horizon = "Quan sát tiếp theo" if IS_AVOCADO else "Quan sát tiếp theo"
    if IS_AVOCADO and window.Date.diff().dropna().eq(pd.Timedelta(days=7)).all():
        horizon = "Tuần tiếp theo · " + (window.Date.iloc[-1] + pd.Timedelta(days=7)).strftime("%d/%m/%Y")
    rows = []
    for framework, column in zip(frameworks, st.columns(len(frameworks))):
        try:
            with st.spinner(f"Đang dự đoán bằng {framework}..."):
                result = predict(window[target].tolist(), cached_bundle(framework, variant), group)
            column.metric("Giá dự đoán" if len(frameworks) == 1 else f"Giá dự đoán · {framework}",
                          f"{result:,.4f}", f"{result-last:+.4f} triệu VND/lượng so với giá cuối")
            column.caption(horizon + f" · {framework} {architecture} · {method}")
            rows.append({"model": framework + " " + architecture, "method": variant, "group": group,
                         "last_observed_date": window.Date.iloc[-1].date(),
                         "last_price_million_vnd_per_luong": last, "prediction_million_vnd_per_luong": result, "horizon": horizon})
        except Exception as exc:
            column.error(f"Không chạy được {framework}: {exc}")
    if rows:
        output = pd.DataFrame(rows)
        if len(rows) == 2:
            st.caption(f"Chênh lệch dự đoán giữa hai model: {abs(rows[0]['prediction_million_vnd_per_luong'] - rows[1]['prediction_million_vnd_per_luong']):,.4f}")
        st.download_button("Tải kết quả CSV", output.to_csv(index=False).encode("utf-8-sig"), f"{TITLE.lower()}-{variant}-forecast.csv", "text/csv")

st.subheader("Tỉ lệ dự đoán đúng và so sánh model · " + architecture + " · " + method)
st.caption("Đánh giá trên tập kiểm thử đã lưu của chuỗi đang chọn. CSV tải lên không làm thay đổi đánh giá này.")
tolerance = st.slider("Ngưỡng sai số được coi là đúng (%)", min_value=1, max_value=30, value=5)
st.caption("Một dự đoán được tính là đúng khi |dự đoán − thực tế| / |thực tế| × 100 ≤ ngưỡng. Đây là tỉ lệ trên dữ liệu lịch sử, không phải xác suất dự đoán tương lai đúng. Mẫu có giá thực tế bằng 0 được loại khỏi tỉ lệ và MAPE.")
evaluation = pd.read_csv(model_dir / "test_predictions.csv")
evaluation = evaluation.loc[evaluation["group"] == group].copy()
if evaluation.empty:
    st.info("Không có mẫu kiểm thử cho chuỗi đang chọn.")
else:
    scores = evaluation_metrics(evaluation, tolerance, architecture)
    scores["Cách dự đoán"] = method
    for (_, row), column in zip(scores.iterrows(), st.columns(2)):
        column.metric(f"{row['Model']} · đúng trong ±{tolerance}%", f"{row['Đúng trong ngưỡng (%)']:.2f}%")
        column.caption(f"{int(row['Mẫu tính tỉ lệ'])} mẫu kiểm thử có giá thực tế khác 0")
    st.dataframe(scores, hide_index=True, width="stretch")
    st.bar_chart(scores.set_index("Model")[["Đúng trong ngưỡng (%)"]])
    evaluation["Date"] = pd.to_datetime(evaluation["Date"])
    st.line_chart(evaluation.set_index("Date")[["actual", f"PyTorch {architecture}", f"Keras {architecture}"]].rename(columns={"actual": "Thực tế"}))
    st.download_button("Tải bảng so sánh CSV", scores.to_csv(index=False).encode("utf-8-sig"), f"{TITLE.lower()}-{variant}-comparison.csv", "text/csv")
with st.expander("Thông số đánh giá toàn bộ tập kiểm thử và huấn luyện"):
    st.caption("MAE, RMSE, MAPE càng thấp càng tốt; R² càng cao càng tốt. Bảng này tính trên toàn bộ tập kiểm thử, gồm tất cả nhóm.")
    st.dataframe(pd.read_csv(model_dir / "model_comparison.csv"), hide_index=True, width="stretch")
