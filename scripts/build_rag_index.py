"""Build a reproducible FAISS index from the intent model's training partition only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json,hashlib,ssl,urllib.request,os,time
import certifi,pandas as pd,numpy as np
from src.rag import EMBEDDING_ID,EMBEDDING_REVISION,INDEX_DIR


def build_index(force=False):
    if (INDEX_DIR/'metadata.json').exists() and not force:
        return json.loads((INDEX_DIR/'metadata.json').read_text())
    import torch,faiss
    from sentence_transformers import SentenceTransformer
    started=time.perf_counter()
    torch.set_num_threads(min(4,os.cpu_count() or 2));faiss.omp_set_num_threads(2)
    raw_path=ROOT/'data/raw/bitext_support.csv'
    raw_path.parent.mkdir(parents=True,exist_ok=True)
    revision='fb86d1b60038970d79fefbfa0c28b47c22d8b961'
    url=f'https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/{revision}/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
    if not raw_path.exists():
        print('Downloading the support dataset...',flush=True)
        with urllib.request.urlopen(url,context=ssl.create_default_context(cafile=certifi.where()),timeout=90) as r:
            content=r.read()
        if hashlib.sha256(content).hexdigest()!='6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434':
            raise ValueError('Unexpected dataset checksum.')
        raw_path.write_bytes(content)
    expected='6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434'
    if hashlib.sha256(raw_path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Cached dataset checksum mismatch.')
    import unicodedata
    def key(s):return ' '.join(unicodedata.normalize('NFC',str(s)).casefold().split())
    train=pd.read_csv(ROOT/'reports/intent/train_split.csv')
    raw=pd.read_csv(raw_path).dropna(subset=['instruction','response','intent'])
    raw['key']=raw.instruction.map(key)
    raw=raw.drop_duplicates('key')
    joined=train[['key','intent']].merge(raw[['key','instruction','response','intent']],on=['key','intent'],validate='one_to_one')
    if len(joined)!=len(train):raise ValueError('Dataset does not match the intent training split.')
    documents=[{'id':hashlib.sha256(r.key.encode()).hexdigest()[:16], 'instruction':r.instruction,
                'response':r.response,'intent':r.intent} for r in joined.itertuples()]
    local_model=ROOT/'models/embedding'
    if local_model.exists():
        encoder=SentenceTransformer(str(local_model),device='cpu')
    else:
        print('Downloading the embedding model...',flush=True)
        encoder=SentenceTransformer(EMBEDDING_ID,revision=EMBEDDING_REVISION,device='cpu',
                                    model_kwargs={'use_safetensors':True})
        encoder.save(str(local_model))
    print(f'Embedding {len(documents):,} training questions. Test and validation questions are excluded.',flush=True)
    embeddings=encoder.encode([d['instruction'] for d in documents],batch_size=64,
                              show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True).astype('float32')
    index=faiss.IndexFlatIP(embeddings.shape[1]);index.add(embeddings)
    INDEX_DIR.mkdir(parents=True,exist_ok=True)
    faiss.write_index(index,str(INDEX_DIR/'index.faiss'))
    (INDEX_DIR/'documents.json').write_text(json.dumps(documents,ensure_ascii=False))
    metadata={'status':'built','documents':len(documents),'dimensions':embeddings.shape[1],
              'embedding_model':EMBEDDING_ID,'embedding_revision':EMBEDDING_REVISION,
              'dataset_url':url,'dataset_sha256':expected,'index':'FAISS IndexFlatIP, normalized vectors',
              'corpus_partition':'intent training split only','build_seconds':time.perf_counter()-started,
              'checksums':{name:hashlib.sha256((INDEX_DIR/name).read_bytes()).hexdigest()
                           for name in ['index.faiss','documents.json']}}
    (INDEX_DIR/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)
    return metadata

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true')
    build_index(parser.parse_args().force)
