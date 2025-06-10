import pandas as pd

MAX_SEQ_LEN = 20  # Chiều dài chuỗi tối đa

def load_data(path):
    print(" Đọc dữ liệu từ file...")
    df = pd.read_csv(path)

    # Giữ lại chỉ các dòng có event_type là 'event' và product_action hợp lệ
    df = df[df['event_type'] == 'event']
    df = df[df['product_action'].isin(['click', 'detail', 'purchase', 'add', 'remove'])]

    # Sắp xếp theo session và thời gian
    df = df.sort_values(by=["session_id_hash", "server_timestamp_epoch_ms"])
    return df

def sample_data(df, target_rows=1_000_000):
    print("Chọn khoảng 1 triệu dòng theo session...")
    session_lengths = df.groupby("session_id_hash").size()
    selected_sessions = []
    total_rows = 0

    for sid, count in session_lengths.items():
        if total_rows + count > target_rows:
            break
        selected_sessions.append(sid)
        total_rows += count

    df_sample = df[df["session_id_hash"].isin(selected_sessions)]
    print(f"Đã chọn {len(df_sample):,} dòng từ {len(selected_sessions):,} session.")
    return df_sample

def build_sequences(df, max_seq_len=MAX_SEQ_LEN):
    print("Tạo chuỗi hành vi từ product_action...")
    grouped = df.groupby("session_id_hash")["product_action"].apply(list)

    X, y = [], []
    for seq in grouped:
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            x_seq = seq[:i]
            if len(x_seq) > max_seq_len:
                x_seq = x_seq[-max_seq_len:] 
            X.append(x_seq)
            y.append(seq[i])

    print(f"Tạo được {len(X):,} cặp chuỗi và nhãn.")
    return X, y

def encode_actions(X, y, max_len=MAX_SEQ_LEN):
    print(" Mã hóa và padding hành vi...")
    action2id = {"click": 0, "detail": 1, "purchase": 2, "add": 3, "remove": 4}
    pad_token = -1

    X_encoded = []
    for seq in X:
        encoded = [action2id[a] for a in seq]
        if len(encoded) < max_len:
            encoded = [pad_token] * (max_len - len(encoded)) + encoded
        X_encoded.append(encoded)

    y_encoded = [action2id[a] for a in y]
    return X_encoded, y_encoded

def save_to_csv(X, y, output_file="processed_data.csv"):
    print(" Lưu kết quả ra file...")
    X_str = [".".join(map(str, seq)) for seq in X]
    df_out = pd.DataFrame({"X": X_str, "y": y})
    df_out.to_csv(output_file, index=False)
    print(f" Đã lưu file: {output_file}")
    print(f" Số dòng trong file sau xử lý: {len(df_out):,}")

if __name__ == "__main__":
    input_file = r"C:/Users/Mias PC/Downloads/shopper_intent_prediction/shopper_intent_prediction/clickstream.csv"  # ← thay bằng tên thật nếu khác
    df = load_data(input_file)
    df_sample = sample_data(df)
    X, y = build_sequences(df_sample)
    X_encoded, y_encoded = encode_actions(X, y)
    save_to_csv(X_encoded, y_encoded)