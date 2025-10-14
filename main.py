from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình CORS (Cross-Origin Resource Sharing)
origins = [
    "http://localhost",
    "http://localhost:5173", #gọi API từ frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Chào mừng bạn đến với FastAPI Backend!"}

@app.get("/api/data")
def get_data():
    return {
        "message": "Dữ liệu được tải thành công từ backend!",
    }
