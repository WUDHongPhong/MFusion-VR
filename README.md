🏆 MFusion-VR: Advanced Multimodal Fusion for Video Retrieval

MFusion-VR (Multi-modal Fusion Video Retrieval) là một hệ thống hỗ trợ người dùng truy xuất hình ảnh đa phương thức nhằm mục đích tìm kiếm nội dung hình ảnh và thông tin video nâng cao từ kho dữ liệu lớn, kết hợp sức mạnh của thị giác máy tính và xử lý ngôn ngữ tự nhiên. Giải pháp này được đánh giá cao và xuất sắc đạt vị trí Quán quân Bảng B tại Hội thi Thử thách Trí tuệ Nhân tạo thành phố Hồ Chí Minh 2025 (AI Challenge 2025 TP.HCM) nhờ khả năng truy vấn ngữ nghĩa chính xác và giao diện thân thiện với người dùng.

🌟 System Overview

Hệ thống được xây dựng trên kiến trúc Hybrid Multimodal Retrieval, tích hợp nhiều phương thức trích xuất đặc trưng và xử lý dữ liệu chuyên sâu để thu hẹp khoảng cách ngữ nghĩa giữa ngôn ngữ tự nhiên và nội dung thị giác.

1. Data Characteristics & Challenges

Dữ liệu được cung cấp bởi Ban tổ chức bao gồm các thành phần cốt lõi sau:

Video Archives: Tập hợp các video thô có nội dung đa dạng và thời lượng ~ 20 phút, đóng vai trò là nguồn dữ liệu gốc cần được truy vấn.

Dense Keyframes: Tập dữ liệu ảnh được trích xuất với tần suất cao từ video gốc. Một video dài có thể bao gồm hàng trăm đến hàng ngàn ảnh, đảm bảo tính liên tục của nội dung nhưng đồng thời tạo áp lực lớn lên khả năng lưu trữ và truy vấn.

Metadata: Các thông tin kỹ thuật đi kèm video như tên, thời gian khung hình (mili giây), fps, frame_idx, link video đến Youtube,….

Còn một số dữ liệu nữa nhưng nhóm không sử dụng đến nên không đề cập.

2. Core Search Functionalities

Hệ thống cung cấp bốn cơ chế truy vấn chính dựa trên các mô hình  tiên tiến:

Text-to-Image Search: Người dùng nhập mô tả văn bản tự nhiên. Hệ thống tính toán độ tương đồng cosine trong không gian nhúng (Embedding Space) và trả về Top-K frames ảnh có điểm số cao nhất.

Image-to-Image Search (Similarity Search): Cho phép người dùng tải lên một hình ảnh mẫu. Hệ thống trích xuất đặc trưng thị giác từ ảnh đó và so khớp với kho dữ liệu vector để tìm kiếm các frame có sự tương đồng về bối cảnh, thực thể hoặc bố cục.

ASR Retrieval (Audio-based): Truy vấn trực tiếp dựa trên lời thoại và âm thanh được trích xuất trực tiếp từ video. Chức năng này giúp xác định các sự kiện thông qua các từ khóa hội thoại.

OCR Retrieval (Frame-based Text): Khác với các hệ thống quét text từ video thô, MFusion-VR tập trung truy vấn chữ viết xuất hiện trực tiếp trong từng frame ảnh (như biển số xe, biển hiệu, văn bản trên đường phố), giúp tăng độ phân giải và độ chính xác khi nhận diện.

3. User Interface & Interaction

Hệ thống được thiết kế với giao diện trực quan, hỗ trợ quy trình làm việc (workflow) từ tìm kiếm đến phân tích kết quả và nộp bài (submission):



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

HF_HOME: Đường dẫn lưu trữ cache của Hugging Face (ví dụ: D:\hf_cache).

GEMINI_API_KEY: Key truy cập từ Google AI Studio dùng cho lớp dịch thuật ngữ nghĩa.

3. Cài đặt Thư viện Lõi

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers datasets accelerate bitsandbytes pandas pyarrow peft flask google-generativeai faiss-cpu Flask-Cors


🧬 Quy trình Xử lý Dữ liệu (Pipeline)

Bước 1: Trích xuất Caption (Image Captioning)

Mỗi thành viên xử lý một tập Keyframes tương ứng bằng cách sử dụng script trích xuất:

python generate_captions.py --target L21


Bước 2: Tối ưu hóa mô hình (Fine-tuning LoRA)

Triển khai huấn luyện trên Google Colab với GPU T4:

Notebook: MFusion-VR Fine-tuning Colab

Phương pháp: Sử dụng file metadata_final.csv sau khi gộp từ các caption để huấn luyện thích nghi mô hình.

Bước 3: Trích xuất Đặc trưng (Feature Extraction)

Tải trọng số đã fine-tune (fine_tuned_model_lora_2025) và tiến hành trích xuất vector lại cho toàn bộ tập dữ liệu để thay thế baseline yếu của BTC.

💡 Các Kỹ thuật Đột phá (Technical Innovations)

Relay Fine-Tuning: Kỹ thuật huấn luyện tiếp sức trên đám mây giúp mô hình học được các thực thể đặc thù trong đô thị Việt Nam nhanh hơn 5-10 lần.

Temporal Interpolation: Thuật toán nội suy tuyến tính dựa trên FPS thực tế giúp ước tính chính xác frame_idx.

Logical Engine (TRAKE.02): Thuật toán giao thoa kết quả giúp xử lý các câu hỏi "giao" của nhiều hành động/đối tượng.

👥 Đội ngũ Phát triển

Dự án MFusion-VR được thực hiện bởi [Tên Team của bạn]. Chúng tôi xin chân thành cảm ơn Ban tổ chức AI Challenge 2025 và UBND TP.HCM.

Lưu ý: Dữ liệu Keyframes và trọng số mô hình thuộc quyền sở hữu của dự án. Vui lòng liên hệ nhóm để biết thêm chi tiết.
