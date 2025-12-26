# 🏆MFusion-VR: Advanced Multimodal Fusion for Video Retrieval

MFusion-VR (Multi-modal Fusion Video Retrieval) là một hệ thống hỗ trợ người dùng truy xuất hình ảnh đa phương thức nhằm mục đích tìm kiếm nội dung hình ảnh và thông tin video nâng cao từ kho dữ liệu lớn, kết hợp sức mạnh của thị giác máy tính và xử lý ngôn ngữ tự nhiên. Giải pháp này được đánh giá cao và xuất sắc đạt vị trí Quán quân Bảng B tại Hội thi Thử thách Trí tuệ Nhân tạo thành phố Hồ Chí Minh 2025 (AI Challenge 2025 TP.HCM) nhờ khả năng truy vấn ngữ nghĩa chính xác và giao diện thân thiện với người dùng.

*Lưu ý: ở đây chỉ cung cấp Pipeline mà nhóm thực hiện và hướng dẫn cách chạy hệ thống dựa trên dữ liệu đã được huấn luyện sẵn.* 

## 🌟 System Overview
Hệ thống được xây dựng trên kiến trúc Hybrid Multimodal Retrieval, tích hợp nhiều phương thức trích xuất đặc trưng và xử lý dữ liệu chuyên sâu để thu hẹp khoảng cách ngữ nghĩa giữa ngôn ngữ tự nhiên và nội dung thị giác.

### 1. Data Characteristics & Challenges

- Dữ liệu được cung cấp bởi Ban tổ chức bao gồm các thành phần cốt lõi sau:

- Video Archives: Tập hợp các video thô có nội dung đa dạng và thời lượng ~ 20 phút, đóng vai trò là nguồn dữ liệu gốc cần được truy vấn.

- Dense Keyframes: Tập dữ liệu ảnh được trích xuất với tần suất cao từ video gốc. Một video dài có thể bao gồm hàng trăm đến hàng ngàn ảnh, đảm bảo tính liên tục của nội dung nhưng tạo áp lực lớn lên khả năng truy xuất.

- Metadata: Các thông tin kỹ thuật đi kèm video như tên, thời gian khung hình (pts_time), fps, frame_idx, link video đến Youtube, v.v.

### 2. Core Search Functionalities

Hệ thống cung cấp bốn cơ chế truy vấn chính dựa trên các mô hình tiên tiến:

- Text-to-Image Search: Người dùng nhập mô tả văn bản tự nhiên. Hệ thống tính toán độ tương đồng cosine trong không gian nhúng (Embedding Space) và trả về Top-K frames ảnh có điểm số cao nhất.

- Image-to-Image Search (Similarity Search): Cho phép người dùng tải lên một hình ảnh mẫu. Hệ thống trích xuất đặc trưng thị giác từ ảnh đó và so khớp với kho dữ liệu vector để tìm kiếm các frame có sự tương đồng.

- ASR Retrieval (Audio-based): Truy vấn dựa trên lời thoại và âm thanh được trích xuất trực tiếp từ video, giúp xác định các sự kiện thông qua từ khóa hội thoại.

- OCR Retrieval (Frame-based Text): Tập trung truy vấn chữ viết xuất hiện trực tiếp trong từng frame ảnh (biển số xe, biển hiệu, văn bản đường phố), giúp tăng độ phân giải và độ chính xác nhận diện.

### 3. User Interface & Interaction (UI/UX)

- Giao diện của MFusion-VR được thiết kế tối ưu để giảm thiểu thời gian tương tác và tăng cường hiệu suất kiểm tra độ chính xác (Verification) trong điều kiện thi đấu:

- Semantic Translation Layer (Gemini Pro): Hệ thống tích hợp Gemini API để tự động chuyển đổi truy vấn từ tiếng Việt sang tiếng Anh trước khi đưa vào mô hình nhúng.

- Scalable Top-K Results: Mặc định hệ thống trả về 100 kết quả hàng đầu, nhưng người dùng có thể tùy chỉnh quy mô (Scale) linh hoạt.

- Lazy Loading Mechanism: Áp dụng kỹ thuật tải chậm thông qua IntersectionObserver. Ảnh chỉ được tải khi người dùng lướt tới khung hình tương ứng, giúp giao diện vận hành mượt mà.

- Integrated Multimedia Player: Trình phát YouTube được tích hợp trực tiếp, cho phép xem video ngay tại mốc thời gian của frame đang chọn mà không cần chuyển tab.

- Real-time Synchronized Tracking: Đồng bộ hóa mốc thời gian video dựa trên thuộc tính pts_time từ metadata, hỗ trợ xác định chính xác thời điểm xảy ra sự kiện theo dữ liệu của BTC.

- Temporal Context (Neighboring Frames): Hiển thị các khung hình lân cận (trước và sau) giúp người dùng hiểu rõ diễn biến sự kiện.


## 🛠 Deployment Guide

### 1. Yêu cầu hệ thống & Môi trường

**Mục tiêu:** Không bị xung đột phiên bản khi chạy hệ thống.

Hệ điều hành: Windows 10/11 hoặc Linux.

Phần cứng: NVIDIA GPU (Tối thiểu 4GB VRAM, khuyên dùng RTX 30-series trở lên).

Python: 3.12.8.

### 2. Tải Dữ liệu và Code

**Mục tiêu:** Có tất cả các file cần thiết từ người quản lý dự án.

**Hành động:** Tải về data và giải nén thư mục dự án (ví dụ: `AIC2025.zip`) vào ổ `D:\`. Sau khi giải nén, bạn sẽ có một thư mục `D:\AIC2025` chứa tất cả dữ liệu thô (`Keyframes`, `objects`, `map-keyframes`, `media-info`,...) và các file code (`.py`, `js`, `css`, `.html`,...).

**📂 Cấu trúc thư mục như sau:**  
```
 D:\AIC2025/
├── Keyframes/                  # Thư mục gốc chứa dữ liệu thị giác.
│   ├── Keyframes_Lxx/          # Các lô dữ liệu lớn (L01, L02...).
│   │   └── Lxx_Vxxx/           # Thư mục video (L05_V002).
│   │       └── 001.jpg         # Các khung hình được đánh số thứ tự.
│   ├── Keyframes_Kxx/          # Các lô dữ liệu lớn (K01, K02...).
│   │   └── Kxx_Vxxx/           # Thư mục video (K12_V022).
│   │       └── 001.jpg         # Các khung hình được đánh số thứ tự.
├── map-keyframes/              # Chứa các file CSV ánh xạ n_frame, pts_time và frame_idx giúp chạy thời gian thực trên UI.
│   ├── Kxx/                    # Subfolder theo batch (K01, K02...)
│   │   └── Kxx_Vxxx.csv        # Cấu trúc: n, pts_time, fps, frame_idx
│   └── Lxx/
│       └── Lxx_Vxxx.csv
├── media-info/                 # Thông tin Metadata chi tiết của video (JSON) giúp nội suy khung hình trực tiếp thực tế lên UI.
│   ├── Kxx/
│   │   └── Kxx_Vxxx.json       # Cấu trúc: author, watch_url, description...
│   └── Lxx/
│       └── Lxx_Vxxx.json
├── asr_result/                 # Dữ liệu lời thoại trích xuất từ âm thanh
│   ├── Kxx/
│   │   └── Kxx_Vxxx.json       # Cấu trúc: segments [start, end, text]
│   └── Lxx/
│       └── Lxx_Vxxx.json
├── ocr.json                    # Tập tin tổng hợp văn bản trích xuất từ toàn bộ Keyframes
├── fine_tuned_model_lora_2025/ # Trọng số mô hình sau khi Fine-tune LoRA
├── app.py                      # Flask Backend - Logic tìm kiếm và Caching
├── index.html                  # Giao diện người dùng chính
├── script.js                   # Logic Frontend và Lazy Loading
├── style.css                   # Định nghĩa phong cách giao diện
├── image_embeddings.npy        # Toàn bộ vector đặc trưng của Keyframes
└── image_paths.json            # Danh sách đường dẫn ảnh ánh xạ với FAISS

```
### 3. Thiết lập môi trường (Environment Variables)

#### 3.1. Thiết lập biến môi trường và API Keys

**Mục tiêu:** Cấu hình để các model tải về được lưu trên ổ D và thiết lập các API key cần thiết.

 **Tạo thư mục Cache trên ổ D:**
 1. Trong terminal PowerShell, gõ lệnh sau và nhấn Enter:
      ```powershell
      mkdir D:\hf_cache
      ```

2.  **Thiết lập Biến Môi trường Hệ thống:**
    Nhấn phím `Windows` và gõ "Edit the system environment variables" rồi chọn kết quả tương ứng.
    
      Trong cửa sổ hiện ra, nhấn nút `Environment Variables...`.
    
      Trong phần `User variables`, nhấn `New...` và tạo 2 biến sau:
    
      **Biến 1 (Hugging Face Cache):**
    
        Variable name: HF_HOME
        Variable value: D:\hf_cache
      **Biến 2 (Gemini API):**
    
        Lấy API Key của bạn từ [Google AI Studio](https://aistudio.google.com/app/apikey).
        Variable name: GEMINI_API_KEY
        Variable value: (Dán chuỗi API key của bạn vào đây)
    
      Nhấn `OK` trên tất cả các cửa sổ để lưu lại.

     *Lưu ý: Đóng VSCode hoàn toàn và mở lại để các thay đổi có hiệu lực.*

3.  **Đăng nhập Hugging Face:**
    Lấy Access Token (quyền `read`) của bạn từ [Hugging Face Tokens](https://huggingface.co/settings/tokens).
    
     Mở lại terminal trong VSCode tại `D:\AIC2025`.
    
    Gõ lệnh sau và nhấn Enter:
      ```powershell
      huggingface-cli login
      ```
      Dán token của bạn vào và nhấn Enter.

      *Lưu ý: khi dán token vào terminal nó sẽ vô hình đoạn mã nên cứ dán rồi ấn Enter.*

#### 3.2. Tạo môi trường ảo và cài đặt thư viện

**Mục tiêu:** Tạo một không gian riêng cho dự án và cài đặt tất cả các công cụ cần thiết.

**Tạo môi trường ảo:** 
Trong terminal, gõ lệnh sau và nhấn Enter (Hoặc có thể tạo bằng Anaconda nhưng phải đúng phiên bản Python):

      powershell
      python -m venv .venv
 

**Kích hoạt môi trường ảo:**
    Gõ lệnh sau và nhấn Enter. Bạn phải làm điều này mỗi khi mở một terminal mới cho hệ thống.
    
      .\.venv\Scripts\Activate.ps1

**Cài đặt các thư viện cần thiết**

    pip install -r requirements.txt
    
## ▶️ Chạy ứng dụng
```
powershell
python app.py
```

## 🧬 Quy trình xử lý Dữ liệu (Pipeline)


## 👥 Đội ngũ Phát triển

Dự án MFusion-VR được thực hiện bởi WuDButterflies (Nguyễn Thành Luân, Nguyễn Xuân Huy, Nguyễn Xuân Hoàng, Khúc Thế Hồng Phong, Nguyễn Hoàng Phúc). Chúng tôi xin chân thành cảm ơn Ban tổ chức AI Challenge TP.HCM 2025 đã tạo điều kiện cho chúng tôi thực hiện dự án này.

## 🔗 Liên kết
Bộ dữ liệu:: 

⭐ Star this repository if it helped you! ⭐
