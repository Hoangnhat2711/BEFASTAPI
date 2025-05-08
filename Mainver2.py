from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các origins trong môi trường development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    timestamp: Optional[str] = None
    sender: str = "bot"
    message: Optional[str] = None

@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root():
    return {"message": "DUE LLM Chat API is running (full context)"}

@app.post("/query", response_model=QueryResponse)
async def process_query(query_request: QueryRequest):
    # Đưa toàn bộ context vào prompt
    system_prompt = (
        "Bạn là một trợ lý AI của trường Đại học Kinh tế - Đại học Đà Nẵng. "
        "Hãy trả lời câu hỏi dựa trên thông tin context dưới đây. "
        "Nếu không tìm thấy thông tin, hãy nói bạn không có dữ liệu phù hợp. Trả lời bằng tiếng Việt tự nhiên."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{full_context}\n\nCâu hỏi: {query_request.query.strip()}"}
    ]
    # Gọi OpenAI Chat Completion
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Lỗi khi gọi LLM: {str(e)}"
    timestamp = datetime.now().strftime("%H:%M:%S")
    return QueryResponse(answer=answer, timestamp=timestamp, sender="bot", message=answer)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 