🏆 MFusion-VR: Advanced Multimodal Fusion for Video Retrieval

MFusion-VR (Multi-modal Fusion Video Retrieval) là một hệ thống hỗ trợ người dùng truy xuất hình ảnh đa phương thức nhằm mục đích tìm kiếm nội dung hình ảnh và thông tin video nâng cao từ kho dữ liệu lớn, kết hợp sức mạnh của thị giác máy tính và xử lý ngôn ngữ tự nhiên. Giải pháp này được đánh giá cao và xuất sắc đạt vị trí Quán quân Bảng B tại Hội thi Thử thách Trí tuệ Nhân tạo thành phố Hồ Chí Minh 2025 (AI Challenge 2025 TP.HCM) nhờ khả năng truy vấn ngữ nghĩa chính xác và giao diện thân thiện với người dùng.

🌟 System Overview

Hệ thống được xây dựng trên kiến trúc Hybrid Multimodal Retrieval, tích hợp nhiều phương thức trích xuất đặc trưng và xử lý dữ liệu chuyên sâu để thu hẹp khoảng cách ngữ nghĩa giữa ngôn ngữ tự nhiên và nội dung thị giác.

1. Data Characteristics & Challenges

Dữ liệu được cung cấp bởi Ban tổ chức bao gồm các thành phần cốt lõi sau:

Video Archives: Tập hợp các video thô có nội dung đa dạng và thời lượng ~ 20 phút, đóng vai trò là nguồn dữ liệu gốc cần được truy vấn.

Dense Keyframes: Tập dữ liệu ảnh được trích xuất với tần suất cao từ video gốc. Một video dài có thể bao gồm hàng trăm đến hàng ngàn ảnh, đảm bảo tính liên tục của nội dung nhưng tạo áp lực lớn lên khả năng truy xuất.

Metadata: Các thông tin kỹ thuật đi kèm video như tên, thời gian khung hình (pts_time), fps, frame_idx, link video đến Youtube, v.v.

2. Core Search Functionalities

Hệ thống cung cấp bốn cơ chế truy vấn chính dựa trên các mô hình tiên tiến:

Text-to-Image Search: Người dùng nhập mô tả văn bản tự nhiên. Hệ thống tính toán độ tương đồng cosine trong không gian nhúng (Embedding Space) và trả về Top-K frames ảnh có điểm số cao nhất.

Image-to-Image Search (Similarity Search): Cho phép người dùng tải lên một hình ảnh mẫu. Hệ thống trích xuất đặc trưng thị giác từ ảnh đó và so khớp với kho dữ liệu vector để tìm kiếm các frame có sự tương đồng.

ASR Retrieval (Audio-based): Truy vấn dựa trên lời thoại và âm thanh được trích xuất trực tiếp từ video, giúp xác định các sự kiện thông qua từ khóa hội thoại.

OCR Retrieval (Frame-based Text): Tập trung truy vấn chữ viết xuất hiện trực tiếp trong từng frame ảnh (biển số xe, biển hiệu, văn bản đường phố), giúp tăng độ phân giải và độ chính xác nhận diện.

3. User Interface & Interaction (UI/UX)

Giao diện của MFusion-VR được thiết kế tối ưu để giảm thiểu thời gian tương tác và tăng cường hiệu suất kiểm tra độ chính xác (Verification) trong điều kiện thi đấu:

Semantic Translation Layer (Gemini Pro): Hệ thống tích hợp Gemini API để tự động chuyển đổi truy vấn từ tiếng Việt sang tiếng Anh trước khi đưa vào mô hình nhúng.

Scalable Top-K Results: Mặc định hệ thống trả về 100 kết quả hàng đầu, nhưng người dùng có thể tùy chỉnh quy mô (Scale) linh hoạt.

Lazy Loading Mechanism: Áp dụng kỹ thuật tải chậm thông qua IntersectionObserver. Ảnh chỉ được tải khi người dùng lướt tới khung hình tương ứng, giúp giao diện vận hành mượt mà.

Integrated Multimedia Player: Trình phát YouTube được tích hợp trực tiếp, cho phép xem video ngay tại mốc thời gian của frame đang chọn mà không cần chuyển tab.

Real-time Synchronized Tracking: Đồng bộ hóa mốc thời gian video dựa trên thuộc tính pts_time từ metadata, hỗ trợ xác định chính xác thời điểm xảy ra sự kiện theo dữ liệu của BTC.

Temporal Context (Neighboring Frames): Hiển thị các khung hình lân cận (trước và sau) giúp người dùng hiểu rõ diễn biến sự kiện.

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

GEMINI_API_KEY: Key truy cập từ Google AI Studio.

3. Cài đặt Thư viện Lõi

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers datasets accelerate bitsandbytes pandas pyarrow peft flask google-generativeai faiss-cpu Flask-Cors


🧬 Quy trình Xử lý Dữ liệu (Pipeline)

Bước 1: Trích xuất Caption (Image Captioning)

Mỗi thành viên xử lý một tập Keyframes tương ứng:

python generate_captions.py --target L21


Bước 2: Tối ưu hóa mô hình (Fine-tuning LoRA)

Triển khai huấn luyện trên Google Colab với GPU T4:

Notebook: MFusion-VR Fine-tuning Colab

Phương pháp: Sử dụng file metadata_final.csv sau khi gộp từ các caption để huấn luyện thích nghi mô hình.

Bước 3: Trích xuất Đặc trưng (Feature Extraction)

Tải trọng số đã fine-tune (fine_tuned_model_lora_2025) và trích xuất lại vector để thay thế baseline cũ của BTC.

💡 Các Kỹ thuật Đột phá (Technical Innovations)

Relay Fine-Tuning: Kỹ thuật huấn luyện tiếp sức trên đám mây giúp mô hình học các thực thể đặc thù nhanh hơn 5-10 lần.

Dual-Layer Caching Strategy: Đây là giải pháp tối ưu hóa hiệu năng cốt lõi của hệ thống, bao gồm:

Disk Caching (hf_cache): Điều hướng lưu trữ các tài nguyên mô hình dung lượng lớn (CLIP weights, LoRA adapters) vào thư mục chỉ định qua biến HF_HOME. Kỹ thuật này giúp giải phóng phân vùng hệ thống, tránh việc tải lại mô hình từ Internet và đảm bảo tính sẵn sàng cao của tài nguyên vật lý.

In-Memory Metadata Caching: Do Metadata cần truy xuất lặp đi lặp lại với tần suất cực cao cho các tính năng thời gian thực, hệ thống thực hiện nạp sẵn toàn bộ dữ liệu vào RAM khi khởi chạy. Giải pháp này triệt tiêu hoàn toàn độ trễ đọc file từ ổ đĩa (Disk I/O), đảm bảo dữ liệu được truyền lên UI ngay lập tức mà không gây nghẽn cổ chai.

Temporal Interpolation: Thuật toán nội suy tuyến tính dựa trên FPS thực tế giúp ước tính chính xác frame_idx.

Logical Engine (TRAKE.02): Thuật toán giao thoa kết quả giúp xử lý các câu hỏi "giao" của nhiều hành động/đối tượng.

👥 Đội ngũ Phát triển

Dự án MFusion-VR được thực hiện bởi [Tên Team của bạn]. Chúng tôi xin chân thành cảm ơn Ban tổ chức AI Challenge 2025 và UBND TP.HCM.

Lưu ý: Dữ liệu Keyframes và trọng số mô hình thuộc quyền sở hữu của dự án. Vui lòng liên hệ nhóm để biết thêm chi tiết.
