"""
FastAPI Application for Quiz Generation and Course Assistance
Uses Gemini 2.5 Flash for multimodal understanding (text, images, videos)
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
import tempfile
from dotenv import load_dotenv
from pathlib import Path
import json
import mimetypes

# Load .env from the same directory as this file (agent folder)
load_dotenv(Path(__file__).resolve().parent / ".env")

# Initialize FastAPI app
app = FastAPI(
    title="Quiz Generator & Course Assistant API",
    description="API for generating quizzes and assisting with course content using Gemini 2.5 Flash",
    version="1.0.0"
)

# Configure Gemini API
# Make sure to set GEMINI_API_KEY environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 2.5 Flash - the most capable multimodal model
MODEL_NAME = "gemini-2.5-flash"

# Pydantic models for request/response
class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: Optional[str] = None

class QuizResponse(BaseModel):
    topic: str
    total_questions: int
    questions: List[QuizQuestion]

class AssistRequest(BaseModel):
    query: str

class AssistResponse(BaseModel):
    query: str
    answer: str

class EvaluateRequest(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class EvaluateResponse(BaseModel):
    question: str
    correct_answer: str
    explanation: str


def upload_to_gemini(file_path: str, mime_type: str = None):
    """
    Upload a file to Gemini File API for processing.
    This is needed for larger files (>20MB) or video files.
    """
    if mime_type is None:
        mime_type = mimetypes.guess_type(file_path)[0]
    
    file = genai.upload_file(file_path, mime_type=mime_type)
    return file


def process_uploaded_files(files: List[UploadFile]) -> tuple[List, List[str]]:
    """
    Process uploaded files and prepare them for Gemini API.
    Returns tuple of (file_parts, temp_file_paths)
    """
    file_parts = []
    temp_paths = []
    
    for uploaded_file in files:
        # Create a temporary file
        suffix = Path(uploaded_file.filename).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        
        # Write uploaded file to temp location
        content = uploaded_file.file.read()
        temp_file.write(content)
        temp_file.close()
        
        temp_paths.append(temp_path)
        
        # Determine MIME type
        mime_type = uploaded_file.content_type or mimetypes.guess_type(uploaded_file.filename)[0]
        
        # For video files or large files, upload to Gemini File API
        if mime_type and (mime_type.startswith('video/') or len(content) > 20 * 1024 * 1024):
            gemini_file = upload_to_gemini(temp_path, mime_type)
            file_parts.append(gemini_file)
        else:
            # For smaller files (images, text, PDFs), can be sent inline
            with open(temp_path, 'rb') as f:
                file_content = f.read()
            
            if mime_type and mime_type.startswith('image/'):
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(file_content))
                file_parts.append(img)
            elif mime_type == 'application/pdf':
                file_parts.append({
                    'mime_type': mime_type,
                    'data': file_content
                })
            else:
                # Text files
                try:
                    text_content = file_content.decode('utf-8')
                    file_parts.append(text_content)
                except:
                    # If can't decode as text, upload as file
                    gemini_file = upload_to_gemini(temp_path, mime_type)
                    file_parts.append(gemini_file)
    
    return file_parts, temp_paths


def cleanup_temp_files(temp_paths: List[str]):
    """Clean up temporary files"""
    for path in temp_paths:
        try:
            os.unlink(path)
        except:
            pass


@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(
    files: List[UploadFile] = File(..., description="Course materials (documents, images, videos)"),
    num_questions: int = Form(5, description="Number of quiz questions to generate"),
    difficulty: str = Form("medium", description="Difficulty level: easy, medium, hard")
):
    """
    Generate quiz questions from uploaded course materials.
    
    Accepts multiple files including:
    - Documents (PDF, TXT, DOCX)
    - Images (JPG, PNG, etc.)
    - Videos (MP4, AVI, MOV, etc.)
    
    Returns a structured JSON with quiz questions, options, and answers.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    temp_paths = []
    
    try:
        # Process uploaded files
        file_parts, temp_paths = process_uploaded_files(files)
        
        # Create the prompt for quiz generation
        prompt = f"""
        You are an expert educational content creator. Analyze the provided course materials 
        (which may include text documents, images, and videos) and generate {num_questions} 
        quiz questions at {difficulty} difficulty level.
        
        For each question:
        1. Create a clear, specific question based on the content
        2. Provide 4 multiple choice options (A, B, C, D)
        3. Indicate the correct answer
        4. Include a brief explanation of why that answer is correct
        
        Return your response in the following STRICT JSON format:
        {{
            "topic": "Main topic of the course material",
            "total_questions": {num_questions},
            "questions": [
                {{
                    "question": "Question text here?",
                    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                    "correct_answer": "A) Option 1",
                    "explanation": "Brief explanation of the correct answer"
                }}
            ]
        }}
        
        Ensure the response is valid JSON only, with no additional text before or after.
        """
        
        # Initialize the model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Create content parts: files first, then prompt
        content_parts = file_parts + [prompt]
        
        # Generate content
        response = model.generate_content(
            content_parts,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        # Parse the response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        quiz_data = json.loads(response_text)
        
        return JSONResponse(content=quiz_data)
    
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse AI response as JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating quiz: {str(e)}"
        )
    finally:
        cleanup_temp_files(temp_paths)


@app.post("/assist", response_model=AssistResponse)
async def assist_with_course(body: AssistRequest):
    """
    Answer a subject-related query. No files required.
    Pass a query and get a clear, educational response.
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        query_escaped = body.query.replace('"', '\\"')
        prompt = f"""
        You are a knowledgeable tutor. Answer the following query in a clear, accurate, and educational manner.
        Be concise but thorough. If the query is about a subject (e.g. math, history, science), explain as appropriate.

        Query: {body.query}

        Return your response in the following STRICT JSON format only, with no extra text before or after:
        {{
            "query": "{query_escaped}",
            "answer": "Your detailed answer here"
        }}
        """
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
        )
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        assist_data = json.loads(response_text)
        return JSONResponse(content=assist_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_answer(body: EvaluateRequest):
    """
    Given a question, multiple choice options, and the correct answer,
    returns an explanation of why that answer is correct.
    """
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    if not body.options or len(body.options) < 2:
        raise HTTPException(status_code=400, detail="options must have at least 2 choices")
    opts_normalized = [o.strip() for o in body.options]
    correct_normalized = body.correct_answer.strip() if body.correct_answer else ""
    if not correct_normalized or correct_normalized not in opts_normalized:
        raise HTTPException(
            status_code=400,
            detail="correct_answer must match one of the provided options (after trimming)"
        )

    try:
        options_text = "\n".join(f"- {opt}" for opt in body.options)
        question_escaped = body.question.replace('"', '\\"')
        correct_escaped = body.correct_answer.replace('"', '\\"')
        prompt = f"""
        You are an educational expert. Given this multiple choice question and its correct answer, explain clearly why that answer is correct.

        Question: {body.question}
        Options:
        {options_text}
        Correct answer: {body.correct_answer}

        Provide a concise but clear explanation (why this answer is right; you may briefly mention why others are wrong if helpful).
        Return your response in the following STRICT JSON format only, with no extra text before or after:
        {{
            "question": "{question_escaped}",
            "correct_answer": "{correct_escaped}",
            "explanation": "Your explanation here"
        }}
        """
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        eval_data = json.loads(response_text)
        return JSONResponse(content=eval_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")


@app.head("/health")
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    Supports both HEAD and GET methods.
    """
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "api_version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Quiz Generator & Course Assistant API",
        "endpoints": {
            "/generate-quiz": "POST - Generate quiz questions from course materials",
            "/assist": "POST - Answer a subject-related query (body: { query })",
            "/evaluate": "POST - Explain why an answer is correct (body: question, options, correct_answer)",
            "/health": "HEAD/GET - Health check endpoint"
        },
        "model": MODEL_NAME,
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
