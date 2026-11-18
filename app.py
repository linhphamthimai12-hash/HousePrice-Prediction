import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Cấu hình trang
st.set_page_config(
    page_title="Dự đoán giá nhà - Nhóm 4",
    page_icon="🏠",
    layout="centered"
)

# Tiêu đề và thông tin nhận diện nhóm
st.title("🏠 ỨNG DỤNG DỰ ĐOÁN GIÁ NHÀ")
st.caption("Thực hiện: **Nhóm 4** | Học phần: **Trí tuệ nhân tạo và ứng dụng** (UTH)")
st.markdown(
    "Nhập thông tin cơ bản của căn nhà. Ứng dụng sử dụng mô hình **Support Vector Machine (SVM - RBF Kernel)** "
    "được huấn luyện từ bộ dữ liệu **`Nhom4_DuĐoanGiaNha.xlsx`** để ước tính giá nhà."
)

# Hàm tải dữ liệu và huấn luyện mô hình (dùng cache để app chạy mượt)
@st.cache_resource
def load_and_train_model():
    # Tự động tìm file dữ liệu Excel hoặc CSV trong repo GitHub
    possible_files = [
        "Nhom4_DuĐoanGiaNha.xlsx",
        "Nhom4_DuDoanGiaNha.xlsx",
        "house_prices_dataset.csv",
        "HousePrice.xlsx"
    ]
    
    df = None
    used_file = ""
    for file in possible_files:
        if os.path.exists(file):
            if file.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
            used_file = file
            break

    if df is None:
        return None, None, None, None, None

    # Làm sạch dữ liệu cơ bản giống bước tiền xử lý trong báo cáo
    df = df.dropna().drop_duplicates()
    
    # Lọc các dòng giá nhà > 0 (nếu có giá trị âm/bằng 0 bất thường)
    if "price" in df.columns:
        df = df[df["price"] > 0]

    features = ["square_feet", "num_rooms", "age", "distance_to_city(km)"]
    target = "price"

    X = df[features]
    y = df[target]

    # Chia tập dữ liệu 80% train - 20% test (random_state=42 giống báo cáo)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Chuẩn hóa dữ liệu bằng StandardScaler cho thuật toán SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình SVR (kernel='rbf')
    svm_model = SVR(kernel="rbf")
    svm_model.fit(X_train_scaled, y_train)

    # Tính các chỉ số đánh giá trên tập test
    y_pred = svm_model.predict(X_test_scaled)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "n_samples": len(df),
        "file_name": used_file
    }

    return svm_model, scaler, features, metrics, df

model, scaler, features, metrics, df = load_and_train_model()

if model is None:
    st.error("⚠️ Không tìm thấy file dữ liệu (`Nhom4_DuĐoanGiaNha.xlsx` hoặc `house_prices_dataset.csv`) trên GitHub repository. Vui lòng tải file dữ liệu lên cùng thư mục với `app.py`.")
    st.stop()

st.success(f"✅ Đã tải dữ liệu từ **`Nhom4_DuĐoanGiaNha.xlsx`** và huấn luyện mô hình **SVM** thành công ({metrics['n_samples']:,} căn nhà).")

# Khu vực nhập thông tin căn nhà
st.subheader("Thông tin căn nhà")

col1, col2 = st.columns(2)
with col1:
    square_feet = st.number_input(
        "Diện tích (square_feet)",
        min_value=100.0,
        max_value=50000.0,
        value=1500.95,
        step=50.0,
        format="%.2f"
    )
    num_rooms = st.number_input(
        "Số phòng (num_rooms)",
        min_value=1,
        max_value=50,
        value=3,
        step=1
    )

with col2:
    age = st.number_input(
        "Tuổi căn nhà - năm (age)",
        min_value=0,
        max_value=200,
        value=10,
        step=1
    )
    distance_to_city = st.number_input(
        "Khoảng cách đến trung tâm - km (distance_to_city)",
        min_value=0.0,
        max_value=200.0,
        value=15.37,
        step=0.5,
        format="%.2f"
    )

# Nút dự đoán
if st.button("DỰ ĐOÁN GIÁ NHÀ", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[square_feet, num_rooms, age, distance_to_city]],
        columns=features
    )
    input_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)[0]

    st.subheader("Kết quả dự đoán")
    st.markdown("Giá nhà ước tính")
    st.markdown(f"## **{predicted_price:,.2f}**")
    st.caption("Kết quả là mức giá ước tính theo đơn vị của cột 'price' trong dữ liệu, không phải báo giá giao dịch chính thức.")

# Phần mở rộng thông tin mô hình & dữ liệu
with st.expander("Thông tin về mô hình và dữ liệu"):
    st.markdown(f"""
    - **Thuật toán sử dụng:** Support Vector Machine cho bài toán hồi quy (`SVR` với `kernel='rbf'`).
    - **Tiền xử lý:** Chuẩn hóa đặc trưng bằng `StandardScaler()`, chia tập dữ liệu tỷ lệ **80% Train - 20% Test** (`random_state=42`).
    - **Biến đầu vào (X):** `square_feet`, `num_rooms`, `age`, `distance_to_city(km)`.
    - **Biến đầu ra (Y):** `price`.
    - **Độ chính xác trên tập kiểm tra (20%):**
        - **MAE:** `{metrics['MAE']:,.2f}`
        - **RMSE:** `{metrics['RMSE']:,.2f}`
        - **R² Score:** `{metrics['R2']:.4f}`
    """)
