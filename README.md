🏆 MFusion-VR: Advanced Multimodal Fusion for Video Retrieval



MFusion-VR (Multi-modal Fusion Video Retrieval) là một hệ thống hỗ trợ người dùng truy xuất hình ảnh đa phương thức nhằm mục đích tìm kiếm nội dung hình ảnh và thông tin video nâng cao từ kho dữ liệu lớn, kết hợp sức mạnh của thị giác máy tính và xử lý ngôn ngữ tự nhiên. Giải pháp này được đánh giá cao và xuất sắc đạt vị trí Quán quân Bảng B tại Hội thi Thử thách Trí tuệ Nhân tạo thành phố Hồ Chí Minh 2025 (AI Challenge 2025 TP.HCM) nhờ khả năng truy vấn ngữ nghĩa chính xác và giao diện thân thiện với người dùng.

🌟 System Overview

Hệ thống được xây dựng trên ý tưởng Hybrid Multimodal Retrieval, tích hợp nhiều phương thức trích xuất đặc trưng như Text-to-Image, Image-to-Image, OCR, ASR để thu hẹp khoảng cách ngữ nghĩa giữa ngôn ngữ tự nhiên và nội dung thị giác:

Semantic Visual Search: Sử dụng không gian đặc trưng chung (Joint Embedding Space) thông qua mô hình nền tảng Apple DFN5B-CLIP.

Domain Adaptation: Tối ưu hóa mô hình cho dữ liệu thực tế tại Việt Nam thông qua kỹ thuật Fine-tuning LoRA trên Google Colab.

Logical Engine (TRAKE.02): Thuật toán giao thoa kết quả (Result Intersection) cho phép thực hiện các truy vấn logic phức tạp (ví dụ: tìm giao của nhiều thực thể/hành động).

Contextual Analysis: Tích hợp nhận dạng ký tự (OCR) và lời thoại (ASR) để làm giàu ngữ cảnh tìm kiếm.

🚀 Technical Achievements

Vị trí: Quán quân (Bảng B) - HCMC AI Challenge 2025.

Hiệu năng: Tốc độ phản hồi < 1s trên hàng triệu frame ảnh nhờ chỉ mục FAISS IndexFlatIP.

Độ chính xác: Tối ưu hóa vượt trội thông qua quy trình huấn luyện "tiếp sức" (Relay Training) trên môi trường GPU điện toán đám mây.

🛠 Deployment Guide

1. Yêu cầu Hệ thống & Môi trường

Hệ điều hành: Windows 10/11 hoặc Linux.

Phần cứng: NVIDIA GPU (Tối thiểu 4GB VRAM, khuyên dùng RTX 30-series trở lên).

Python: 3.12.

2. Thiết lập Biến Môi trường (Environment Variables)

Để hệ thống vận hành ổn định và quản lý cache hiệu quả trên ổ đĩa có dung lượng lớn:

HF_HOME: Đường dẫn lưu trữ cache của Hugging Face (ví dụ: D:\hf_cache).

GEMINI_API_KEY: Key truy cập từ Google AI Studio dùng cho lớp dịch thuật ngữ nghĩa.

3. Cài đặt Thư viện Lõi

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers datasets accelerate bitsandbytes pandas pyarrow peft flask google-generativeai faiss-cpu Flask-Cors


🧬 Quy trình Xử lý Dữ liệu (Pipeline)

Quy trình được thiết kế theo dạng Modular & Parallel, cho phép các thành viên trong nhóm xử lý dữ liệu song song trước khi gộp lại.

Bước 1: Trích xuất Caption (Image Captioning)

Mỗi thành viên xử lý một tập Keyframes tương ứng bằng cách sử dụng script trích xuất:

python generate_captions.py --target L21


Bước 2: Tối ưu hóa mô hình (Fine-tuning LoRA)

Do giới hạn tài nguyên cục bộ, chúng tôi triển khai huấn luyện trên Google Colab với GPU T4 để đảm bảo tốc độ và hiệu suất:

Notebook: MFusion-VR Fine-tuning Colab

Phương pháp: Sử dụng file metadata_final.csv sau khi gộp từ các caption để huấn luyện thích nghi mô hình với miền dữ liệu.

Bước 3: Trích xuất Đặc trưng (Feature Extraction)

Tải trọng số đã fine-tune (fine_tuned_model_lora_2025) về máy cục bộ và tiến hành trích xuất vector cho toàn bộ tập dữ liệu:

python extract_features_modular.py --target L21


Bước 4: Tích hợp & Vận hành (Indexing & Hosting)

Gộp các file vector thành image_embeddings.npy và image_paths.json.

Khởi chạy Backend server:

python app.py


Truy cập UI tại: http://localhost:5000.

💡 Các Kỹ thuật Đột phá (Technical Innovations)

Relay Fine-Tuning: Kỹ thuật huấn luyện tiếp sức trên đám mây giúp mô hình học được các thực thể đặc thù trong đô thị Việt Nam (biển báo, loại phương tiện...) nhanh hơn 5-10 lần so với máy trạm đơn lẻ.

Temporal Interpolation: Thuật toán nội suy tuyến tính dựa trên FPS thực tế giúp ước tính chính xác frame_idx từ thời gian phát, đáp ứng yêu cầu khắt khe về độ chính xác thời gian của Ban Tổ Chức.

Semantic Translation Layer: Sử dụng Gemini Pro để tinh chỉnh các truy vấn Tiếng Việt phức tạp sang Anh ngữ chuẩn xác trước khi đưa vào không gian nhúng của CLIP.

👥 Đội ngũ Phát triển

Dự án MFusion-VR được thực hiện bởi [Tên Team của bạn]. Chúng tôi xin chân thành cảm ơn Ban tổ chức AI Challenge 2025 và UBND TP.HCM đã tạo cơ hội cho những tài năng công nghệ tỏa sáng.

Lưu ý: Dữ liệu Keyframes và trọng số mô hình thuộc quyền sở hữu của dự án. Vui lòng liên hệ nhóm để biết thêm chi tiết.
