import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Banner và CSS custom
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #ff5858 0%, #f09819 100%);
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        padding: 0.5em 2em;
        box-shadow: 0 4px 14px 0 rgba(255, 88, 88, 0.2);
    }
    .stAlert {
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
    <div style='text-align:center; margin-bottom: 10px;'>
        <img src='https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3ZmVyY2RxbzNqenU0NTIzNTd6azFwcDVoZ3RnN3dhdGo3eDhvNmhocCZlcD12MV9naWZzX3JlbGF0ZWQmY3Q9Zw/MdMCVvLjbyyUhUeClf/giphy.gif' width='120'>
        <h1 style='color:#ff5858; font-size: 2.8em; font-family:Comic Sans MS, cursive, sans-serif;'>
            🌈🧠 Dự đoán hành vi người dùng tiếp theo (LSTM) 🎉✨
        </h1>
        <img src='https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif' width='120'>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    "<span style='color:#00bfff;font-size:20px;font-weight:bold;'>Chọn chuỗi hành vi từ dữ liệu thực tế hoặc nhập chuỗi mới (cách nhau bằng dấu phẩy, hỗ trợ cả tiếng Việt lẫn tiếng Anh) 😎👇</span>",
    unsafe_allow_html=True
)

# Mã hóa hành vi
action2id = {
    "chọn": 0,"click": 0,
    "xem thêm": 1,"detail": 1,
    "mua": 2, "purchase": 2,
    "thêm": 3, "add": 3,
    "xóa": 4, "remove": 4
}
id2action = {v: k for k, v in action2id.items() if v in [0,1,2,3,4]}

MAX_LEN = 20
PAD_TOKEN = -1

# Load mô hình
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


sample_seq = st.selectbox(
    "🌟 Chọn một chuỗi hành vi mẫu từ dữ liệu:",
    df["X"].head(1000),  # chỉ lấy 1000 chuỗi đầu cho nhanh
    index=0
)

# Hoặc nhập chuỗi mới
input_seq = st.text_input("💬 Hoặc nhập chuỗi hành vi: (ex: click/chọn/0, detail/xem thêm/1, purchase/mua/2, add/thêm/3, remove/xóa/4)", sample_seq)

if st.button("✨📈 Dự đoán ngay!"):
    try:
        # Tự động nhận biết dấu phân cách (dấu phẩy hoặc dấu chấm)
        if "." in input_seq and not "," in input_seq:
            sep = "."
        else:
            sep = ","
        # Nếu nhập chuỗi dạng số (từ file), chuyển về tên hành vi
        if all(x.strip().lstrip('-').isdigit() for x in input_seq.replace('.', ',').split(',')):
            actions = []
            for i in input_seq.replace('.', ',').split(','):
                i = i.strip()
                if i not in ['', '-1']:
                    idx = int(i)
                    if idx in id2action:
                        actions.append(id2action[idx])
        else:
            actions = [a.strip().lower() for a in input_seq.split(sep) if a.strip().lower() in action2id]

        st.markdown(
            f"<span style='color:#ff5858;font-size:18px;'>🟢 <b>Actions:</b> {actions}</span>",
            unsafe_allow_html=True
        )

        if not actions:
            st.warning("⚠️ Dữ liệu bạn nhập không phù hợp hoặc không chứa hành động hợp lệ. Vui lòng kiểm tra lại! Hãy nhập các chuỗi hành động như click, add, detail,... hoặc xem thêm, chọn, mua,...")
        else:
            encoded = [action2id[a] for a in actions]
            if len(encoded) > MAX_LEN:
                encoded = encoded[-MAX_LEN:]
            elif len(encoded) < MAX_LEN:
                encoded = [PAD_TOKEN] * (MAX_LEN - len(encoded)) + encoded

            X = np.array(encoded).reshape(1, -1)
            # Hiển thị input vào model dưới dạng bảng
            st.markdown("#### 🔢 Input vào model:")
            st.dataframe(pd.DataFrame(X, columns=[f"{i}" for i in range(X.shape[1])]))

            prediction = model.predict(X)
            # Hiển thị output model dưới dạng bảng
            st.markdown("#### 📊 Output model:")
            st.dataframe(pd.DataFrame(prediction, columns=[str(i) for i in range(prediction.shape[1])]))

            pred_class = np.argmax(prediction, axis=1)[0]
            true_label = label_classes[pred_class]
            st.markdown(
                f"""
                <div style='background:#eaffd0;padding:20px;border-radius:10px;margin-top:20px;text-align:center;'>
                    <span style='font-size:28px;color:#008000;'>🌟👉 Hành vi tiếp theo dự đoán: <b>{id2action[true_label]}</b> 🎉</span>
                    <br>
                    <img src='https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExYWhsMXg2dXk5a3I1NWljODk5bjlqMjlmcGlob3NxY3Z2ODRzNXlueSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/UtEUhkfriklonVdweC/giphy.gif' width='120'>
                </div>
                """,
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")


