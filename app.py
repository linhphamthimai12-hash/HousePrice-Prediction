import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dự đoán giá nhà", page_icon="🏠", layout="centered")

@st.cache_data
def load_data():
    data = pd.read_excel("HousePrice.xlsx")
    required = ["square_feet", "num_rooms", "age", "distance_to_city(km)", "price"]
    data = data[required].dropna()
    return data

@st.cache_resource
def train_model():
    data = load_data()
    features = ["square_feet", "num_rooms", "age", "distance_to_city(km)"]
    model = LinearRegression()
    model.fit(data[features], data["price"])
    return model

st.title("🏠 ỨNG DỤNG DỰ ĐOÁN GIÁ NHÀ")
st.write(
    "Nhập thông tin cơ bản của căn nhà. Ứng dụng sử dụng mô hình "
    "Linear Regression được huấn luyện từ bộ dữ liệu HousePrice.xlsx "
    "để ước tính giá nhà."
)

try:
    data = load_data()
    model = train_model()

    st.success(f"Đã tải dữ liệu và huấn luyện mô hình thành công ({len(data):,} căn nhà).")

    st.subheader("Thông tin căn nhà")
    col1, col2 = st.columns(2)

    with col1:
        square_feet = st.number_input(
            "Diện tích (square feet)",
            min_value=1.0,
            max_value=float(max(data["square_feet"].max() * 1.5, 2)),
            value=float(data["square_feet"].median()),
            step=50.0
        )
        num_rooms = st.number_input(
            "Số phòng",
            min_value=1,
            max_value=30,
            value=int(data["num_rooms"].median()),
            step=1
        )

    with col2:
        age = st.number_input(
            "Tuổi căn nhà (năm)",
            min_value=0,
            max_value=200,
            value=int(data["age"].median()),
            step=1
        )
        distance = st.number_input(
            "Khoảng cách đến trung tâm (km)",
            min_value=0.0,
            max_value=float(max(data["distance_to_city(km)"].max() * 1.5, 1)),
            value=float(data["distance_to_city(km)"].median()),
            step=0.5
        )

    if st.button("DỰ ĐOÁN GIÁ NHÀ", type="primary", use_container_width=True):
        sample = pd.DataFrame([{
            "square_feet": square_feet,
            "num_rooms": num_rooms,
            "age": age,
            "distance_to_city(km)": distance
        }])
        prediction = float(model.predict(sample)[0])
        st.subheader("Kết quả dự đoán")
        st.metric("Giá nhà ước tính", f"{prediction:,.2f}")
        st.caption("Kết quả là mức giá ước tính theo đơn vị của cột 'price' trong dữ liệu, không phải báo giá giao dịch chính thức.")

    with st.expander("Thông tin về mô hình và dữ liệu"):
        st.write("**Thuật toán:** Linear Regression")
        st.write(f"**Số dòng dữ liệu sử dụng:** {len(data):,}")
        st.write("**Các yếu tố đầu vào:** diện tích, số phòng, tuổi căn nhà và khoảng cách đến trung tâm.")
        st.write("Ứng dụng huấn luyện mô hình từ dữ liệu đi kèm khi khởi chạy.")

except Exception as e:
    st.error("Không thể tải dữ liệu hoặc huấn luyện mô hình. Hãy kiểm tra file HousePrice.xlsx có nằm cùng thư mục với app.py hay không.")
    st.exception(e)
