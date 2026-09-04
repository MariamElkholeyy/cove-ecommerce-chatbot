"""Local FastAPI chat app. Run only on loopback; no public deployment authentication."""
from pathlib import Path
import os,threading,logging
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from src.rag import ROOT,SupportRAG,ServiceError
load_dotenv(ROOT/'.env')
state={'engine':None,'error':None,'loading':True}
key_lock=threading.Lock()

def initialize():
    try:state['engine']=SupportRAG()
    except Exception:
        logging.exception('RAG initialization failed')
        state['error']='A model component could not load. Check RAG_GUIDE.md, rebuild what is missing, and restart the app.'
    finally:state['loading']=False

@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=initialize,daemon=True).start()
    yield

app=FastAPI(title='Cove Support',lifespan=lifespan,docs_url=None,redoc_url=None)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','testserver'])

@app.middleware('http')
async def local_origin(request:Request,call_next):
    if request.method in {'POST','PUT','DELETE'}:
        from urllib.parse import urlparse
        origin=request.headers.get('origin')
        if origin and urlparse(origin).netloc != request.headers.get('host'):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'Cross-origin requests are not allowed.'},status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    response.headers['Cache-Control']='no-store'
    response.headers['Content-Security-Policy']="default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    return response

class HistoryMessage(BaseModel):
    role:str=Field(pattern='^(user|assistant)$')
    content:str=Field(max_length=4000)
class ChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=3000)
    history:list[HistoryMessage]=Field(default_factory=list,max_length=6)
class KeyRequest(BaseModel):
    api_key:str=Field(min_length=20,max_length=200)

@app.get('/api/status')
def status():
    engine=state['engine']
    return {'ready':engine is not None,'loading':state['loading'],'error':state['error'],
            'key_configured':bool(os.environ.get('GROQ_API_KEY')),
            'documents':engine.manifest['documents'] if engine else 0,
            'sentiment_model':engine is not None}

@app.post('/api/key')
def set_key(body:KeyRequest):
    key=body.api_key.strip()
    if not key.startswith('gsk_'):raise HTTPException(400,'Enter a valid Groq API key.')
    # Session-only replacement, never written to HTML, logs, browser storage, or reports.
    with key_lock:os.environ['GROQ_API_KEY']=key
    return {'configured':True,'message':'Key updated for this app session.'}

@app.post('/api/chat')
def chat(body:ChatRequest):
    if not body.message.strip():raise HTTPException(422,'Please type a message.')
    engine=state['engine']
    if engine is None:raise HTTPException(503,state['error'] or 'The knowledge library is still loading. Please try again shortly.')
    try:return engine.answer(body.message,[h.model_dump() for h in body.history])
    except ServiceError as exc:raise HTTPException(exc.status,str(exc)) from None
    except Exception:
        logging.exception('Chat failed')
        raise HTTPException(500,'Something went wrong while preparing the answer. Please try again.') from None

@app.get('/')
def home():return FileResponse(ROOT/'app/static/index.html')
app.mount('/static',StaticFiles(directory=ROOT/'app/static'),name='static')
