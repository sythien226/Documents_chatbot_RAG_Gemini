import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import messages_from_dict
import os
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pandas as pd
from paddleocr import PaddleOCR
from htmlTemplates1 import css, bot_template, user_template
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Khởi tạo PaddleOCR (chỉ 1 lần)
@st.cache_resource
def init_paddle_ocr():
    """Khởi tạo PaddleOCR với cache để tránh reload nhiều lần"""
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        return ocr
    except Exception as e:
        st.error(f"Lỗi khởi tạo PaddleOCR: {str(e)}")
        return None

def detect_scan_pdf(pdf_path):
    """Kiểm tra PDF có phải là scan không bằng cách phân tích nội dung text"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_chars = 0
            for page in pdf.pages[:3]:  # Kiểm tra 3 trang đầu
                text = page.extract_text()
                if text:
                    total_chars += len(text.strip())
            
            # Nếu ít hơn 100 ký tự trong 3 trang đầu => có thể là scan
            return total_chars < 100
    except:
        return False

def extract_tables_from_text_pdf(pdf_path):
    """Trích xuất bảng từ PDF text bằng pdfplumber"""
    tables_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    tables_text += f"\n--- Table of page {page_num + 1} ---\n"
                    for table_num, table in enumerate(tables):
                        if table:
                            # Chuyển table thành DataFrame để format đẹp
                            df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                            tables_text += f"\nTable {table_num + 1}:\n"
                            tables_text += df.to_string(index=False) + "\n\n"
    except Exception as e:
        st.warning(f"Lỗi trích xuất bảng từ PDF text: {str(e)}")
    
    return tables_text

def extract_tables_from_scan_pdf(pdf_path, ocr):
    """Trích xuất bảng từ PDF scan bằng PaddleOCR"""
    tables_text = ""# tạo chuỗi tables_text
    try:
        # Chuyển PDF thành ảnh
        images = pdf2image.convert_from_path(pdf_path, 
                                             dpi=200,# độ phân giải 200200
                                             poppler_path=r"C:/Release-24.08.0-0/poppler-24.08.0/Library/bin"
                                             )
        
        for page_num, image in enumerate(images):# duyệt từng ảnh. page)num là số thứ tự trang
            # Chuyển PIL image thành numpy array
            img_array = np.array(image) # chuyển ảnh sang mảng numpy đẻe xứ lí bằng opencv
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)# numpy-> RGBRGB
            
            # Sử dụng OCR để detect text và box
            result = ocr.ocr(img_cv, cls=True) #gọi hàm OCR để nhận diện chữ
            
            if result and result[0]:# nếu có kq trả về từ oCR thì tiếp tục xử lí
                # Phân tích layout để tìm bảng
                tables_text += f"\n--- Content of page scan {page_num + 1} ---\n"
                
                # Sắp xếp text theo vị trí (top-to-bottom, left-to-right)
                text_boxes = [] # khởi tạo danh sách text box để luuư tọa độ các khối VB
                for line in result[0]:# duyệt từng dòng văn bản trong ảnhảnh
                    if line:# nếu dòng k rỗng
                        box = line[0] # danh sách 4 điểm tọa độ của khung chữchữ
                        text = line[1][0] # nội dung vbvb
                        confidence = line[1][1] # độ tin cậy 0-11
                        
                        if confidence > 0.5:  # Chỉ lấy text có độ tin cậy cao
                            # Tính toán vị trí trung tâm
                            center_y = (box[0][1] + box[2][1]) / 2
                            center_x = (box[0][0] + box[2][0]) / 2
                            text_boxes.append((center_y, center_x, text))
                
                # Sắp xếp theo vị trí
                text_boxes.sort(key=lambda x: (x[0], x[1]))# sx trên xuống , trái -phảiphải
                
                # Ghép text thành đoạn văn
                current_line_y = None
                current_line_texts = [] # danh sách các từ trong dòngdòng
                
                for y, x, text in text_boxes:# lặp qua từng chữ OCR trong ds text_boxbox
                    if current_line_y is None or abs(y - current_line_y) > 20:
                        # Dòng mới
                        if current_line_texts: # nếu đang có nd dòng cũcũ
                            tables_text += " ".join(current_line_texts) + "\n" # ghi vào table texttext
                        current_line_texts = [text]# hết nd dòng cũ thì 
                        current_line_y = y #khởi tạo dòng mới với từ hien taitai
                    else:
                        # Cùng dòng
                        current_line_texts.append(text)
                
                # Thêm dòng cuối
                if current_line_texts:
                    tables_text += " ".join(current_line_texts) + "\n"
                    
    except Exception as e:
        st.warning(f"Lỗi OCR PDF scan: {str(e)}")
    
    return tables_text

def get_pdf_text_enhanced(pdf_docs):
    """Trích xuất text từ PDF với hỗ trợ OCR và table extraction"""
    if not pdf_docs:
        return ""
    #khoi tao OCR và giao diệndiện
    ocr = init_paddle_ocr() #khởi tạo mô hình paddle OCR
    text = ""
    progress_bar = st.progress(0) #dùng để hiển thị tiến độ trên streamlitstreamlit
    status_text = st.empty()# dùng để h thị trạng tháithái
    
    for i, pdf_file in enumerate(pdf_docs):#duyệt qua từng filefile
        try:
            status_text.text(f'Đang phân tích {pdf_file.name}...')
            
            # Lưu file tạm để xử lý
            temp_path = f"temp_{pdf_file.name}"
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Kiểm tra loại PDF
            is_scan = detect_scan_pdf(temp_path)
            
            if is_scan:# nếu là PDF scan dùng OCROCR
                status_text.text(f'🔍 PDF scan detected - Đang OCR {pdf_file.name}...')
                
                # Xử lý PDF scan
                if ocr:
                    scan_text = extract_tables_from_scan_pdf(temp_path, ocr)
                    if scan_text.strip():
                        text += f"\n=== PDF SCAN: {pdf_file.name} ===\n"# thêm tieu đề để đanhs giấu PDF scan
                        text += scan_text + "\n" # nối scan_text vào text chung 
                    else:
                        st.warning(f"Không trích xuất được text từ PDF scan: {pdf_file.name}")
                else:
                    st.error("PaddleOCR không khả dụng!")
                    
            else:
                status_text.text(f'📄 PDF text detected - Đang xử lý {pdf_file.name}...')
                
                # Xử lý PDF text thông thường
                pdf_reader = PdfReader(temp_path)# đọc file PDF dạng text
                regular_text = ""# khởi tạo chuỗi rỗng 
                
                for page_num, page in enumerate(pdf_reader.pages):#lặp qua từng trang của pdf 
                    page_text = page.extract_text() #trichs xuất text từ các trangtrang
                    if page_text:#nếu page_text k rỗng
                        regular_text += f"\n--- Page {page_num + 1} ---\n"# thêm vào regular text vào 1 đoạn chú thích phân trang
                        regular_text += page_text + "\n" # nối page text vào regular_texttext
                
                # Trích xuất bảng từ PDF text
                tables_text = extract_tables_from_text_pdf(temp_path)
                
                if regular_text.strip() or tables_text.strip():#kiểm tra xem có nd nào dc trích ra kk
                    text += f"\n=== PDF TEXT: {pdf_file.name} ===\n"#thêm tiêu đề
                    text += regular_text + tables_text + "\n"# thêm vào biến text tổng 
            
            # Xóa file tạm
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            st.error(f"❌ Lỗi xử lý {pdf_file.name}: {str(e)}")
            continue
            
        progress_bar.progress((i + 1) / len(pdf_docs))
    
    progress_bar.empty()
    status_text.empty()
    
    if not text.strip():
        st.warning("⚠️ Không trích xuất được text từ bất kỳ PDF nào!")
    else:
        st.success(f"✅ Đã xử lý {len(pdf_docs)} file PDF thành công!")
        
        # Hiển thị thống kê
        word_count = len(text.split())
        char_count = len(text)
        st.info(f"📊 Thống kê: {word_count:,} từ, {char_count:,} ký tự")
    
    return text

def get_text_chunks(text):
    """Chia text thành các chunk nhỏ"""
    if not text or not text.strip():
        st.warning("Text rỗng, không thể tạo chunks!")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        #separators=["\n\n", "\n", ".", "!", "?", " "],
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        st.warning("Không thể chia text thành chunks!")
        return []
    
    st.info(f"📝 Đã tạo {len(chunks)} text chunks")
    return chunks

def get_vectorstore(text_chunks, api_key):
    """Tạo vector store từ text chunks"""
    if not text_chunks:
        st.error("Không có text chunks để tạo vector store!")
        return None
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.success("✅ Đã tạo vector store thành công!")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Lỗi tạo vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore, api_key):
    """Tạo conversation chain"""
    if not vectorstore:
        return None
    
    try:
        # prompt_template = """
        # Hãy trả lời câu hỏi một cách chi tiết dựa trên nội dung được cung cấp. 
        # Nếu thông tin có bảng hoặc dữ liệu có cấu trúc, hãy trình bày rõ ràng.
        # Nếu không tìm thấy thông tin trong nội dung, hãy nói "Thông tin không có trong tài liệu".
        
        # Nội dung: {context}
        
        # Câu hỏi: {question}
        
        # Trả lời:
        # """
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", 
            temperature=0.3, 
            google_api_key=api_key
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        return conversation_chain
    except Exception as e:
        st.error(f"❌ Lỗi tạo conversation chain: {str(e)}")
        return None

def handle_userinput_with_manual_history(user_question):
    """Xử lý input của user và lưu lịch sử chat"""
    if "chat_history_dict" not in st.session_state:
        st.session_state.chat_history_dict = []

    # Thêm câu hỏi mới
    st.session_state.chat_history_dict.append({
        "type": "human", 
        "data": {"content": user_question}
    })

    try:
        # Chuyển lịch sử dict thành list message object
        chat_history_messages = messages_from_dict(st.session_state.chat_history_dict)

        # Gọi chain
        response = st.session_state.conversation.invoke({
            "question": user_question,
            "chat_history": chat_history_messages
        })

        answer = response.get("answer", "Xin lỗi, tôi không thể trả lời câu hỏi này.")

        # Thêm câu trả lời vào lịch sử
        st.session_state.chat_history_dict.append({
            "type": "ai", 
            "data": {"content": answer}
        })

    except Exception as e:
        st.error(f"❌ Lỗi xử lý câu hỏi: {str(e)}")
        st.session_state.chat_history_dict.append({
            "type": "ai", 
            "data": {"content": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn."}
        })

    # Hiển thị toàn bộ lịch sử
    for msg in st.session_state.chat_history_dict:
        if msg["type"] == "human":
            st.write(user_template.replace("{{MSG}}", msg["data"]["content"]), 
                    unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg["data"]["content"]), 
                    unsafe_allow_html=True)

def main():
    """Hàm chính của ứng dụng"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Kiểm tra API key
    if not api_key:
        st.error("🔑 Không tìm thấy GOOGLE_API_KEY! Vui lòng kiểm tra file .env.")
        st.stop()

    st.set_page_config(
        page_title="Chat với PDF (OCR + Table)", 
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    # Khởi tạo session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history_dict" not in st.session_state:
        st.session_state.chat_history_dict = []

    # Header
    st.title("📚 Chat với PDF (Hỗ trợ OCR & Bảng)")
    st.markdown("""
    ### 🚀 Tính năng:
    - ✅ PDF thường (text) + trích xuất bảng bằng pdfplumber
    - ✅ PDF scan + OCR bằng PaddleOCR
    - ✅ Tự động phát hiện loại PDF
    - ✅ Chat thông minh với nội dung PDF
    """)

    # Input câu hỏi
    user_question = st.text_input("💬 Đặt câu hỏi về tài liệu của bạn:")

    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput_with_manual_history(user_question)
        else:
            st.warning("⚠️ Vui lòng upload và xử lý PDF trước!")

    # Sidebar
    with st.sidebar:
        st.subheader("📁 Tải lên tài liệu")
        
        pdf_docs = st.file_uploader(
            "Chọn file PDF (hỗ trợ cả PDF text và scan):",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if pdf_docs:
            st.success(f"✅ Đã chọn {len(pdf_docs)} file PDF")
            total_size = sum([pdf.size for pdf in pdf_docs]) / (1024*1024)
            st.info(f"💾 Tổng dung lượng: {total_size:.2f} MB")
        
        if st.button("🔄 Xử lý PDF", type="primary"):
            if pdf_docs:
                with st.spinner("🔄 Đang xử lý PDF..."):
                    # Trích xuất text
                    raw_text = get_pdf_text_enhanced(pdf_docs)
                    
                    if raw_text.strip():
                        # Tạo chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        if text_chunks:
                            # Tạo vectorstore
                            vectorstore = get_vectorstore(text_chunks, api_key)
                            
                            if vectorstore:
                                # Tạo conversation chain
                                st.session_state.conversation = get_conversation_chain(
                                    vectorstore, api_key
                                )
                                
                                if st.session_state.conversation:
                                    # Reset lịch sử chat
                                    st.session_state.chat_history_dict = []
                                    st.success("🎉 Xử lý hoàn tất! Bạn có thể đặt câu hỏi ngay.")
                                else:
                                    st.error("❌ Không thể tạo conversation chain!")
                            else:
                                st.error("❌ Không thể tạo vector store!")
                        else:
                            st.error("❌ Không thể tạo text chunks!")
                    else:
                        st.error("❌ Không trích xuất được text từ PDF!")
            else:
                st.warning("⚠️ Vui lòng chọn ít nhất 1 file PDF!")
        
        # Nút xóa lịch sử
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.chat_history_dict = []
            st.success("✅ Đã xóa lịch sử chat!")
            st.rerun()

if __name__ == '__main__':
    main()
