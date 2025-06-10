import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Mã hóa hành vi
action2id = {"click": 0, "detail": 1, "purchase": 2, "add": 3, "remove": 4}
id2action = {v: k for k, v in action2id.items()}

MAX_LEN = 20
PAD_TOKEN = -1

# Load mô hình GRU
@st.cache_resource
def load_model_file():
    model = load_model("my_lstm_model.h5")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Load dữ liệu đã tiền xử lý (300k dòng)
@st.cache_data
def load_processed_data():
    df = pd.read_csv("processed_data_300k.csv")
    return df

model = load_model_file()
df = load_processed_data()
label_classes = np.unique(df["y"])  # Lấy lại thứ tự nhãn gốc đã dùng khi train

st.title("🧠 Dự đoán hành vi người dùng tiếp theo (LSTM)")
st.markdown("Chọn chuỗi hành vi từ dữ liệu thực tế hoặc nhập chuỗi mới (cách nhau bằng dấu phẩy).")

# Cho phép chọn một chuỗi từ dữ liệu thực tế
sample_seq = st.selectbox(
    "Chọn một chuỗi hành vi mẫu từ dữ liệu:",
    df["X"].head(1000),  # chỉ lấy 1000 chuỗi đầu cho nhanh
    index=0
)

# Hoặc nhập chuỗi mới
input_seq = st.text_input("Hoặc nhập chuỗi hành vi:", sample_seq)

if st.button("📈 Dự đoán"):
    try:
        # Nếu nhập chuỗi dạng số (từ file), chuyển về tên hành vi
        if all(x.strip().isdigit() or x.strip() == '-' for x in input_seq.replace('.', ',').split(',')):
            actions = [id2action[int(i)] for i in input_seq.replace('.', ',').split(',') if i.strip() not in ['', '-1']]
        else:
            actions = [a.strip() for a in input_seq.split(",") if a.strip() in action2id]

        st.write("Actions:", actions)  # Debug xem actions đã đúng chưa

        encoded = [action2id[a] for a in actions]
        if len(encoded) > MAX_LEN:
            encoded = encoded[-MAX_LEN:]
        elif len(encoded) < MAX_LEN:
            encoded = [PAD_TOKEN] * (MAX_LEN - len(encoded)) + encoded

        X = np.array(encoded).reshape(1, -1)
        st.write("Input vào model:", X)  # Kiểm tra input
        prediction = model.predict(X)
        st.write("Output model:", prediction)  # Kiểm tra output
        pred_class = np.argmax(prediction, axis=1)[0]
        true_label = label_classes[pred_class]
        st.success(f"👉 Hành vi tiếp theo dự đoán: **{id2action[true_label]}**")
    except Exception as e:
        st.error(f"Lỗi: {e}")
        
