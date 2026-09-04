"""Exercise the running HTTP app and preserve real results, including failures."""
import sys,json,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_rag import SCENARIOS

if __name__=='__main__':
    reports=ROOT/'reports/rag'
    results=[]
    cases=[(name,message,[]) for name,message in SCENARIOS]
    cases += [('arabic_negative','أنا غاضبة جدا من تأخير طلبي ومحدش بيرد عليا',[])]
    cases += [('followup','What if I cannot find that information?',None)]
    for name,message,history in cases:
        if results:time.sleep(22)
        if history is None:
            tracking=next((r for r in results if r['scenario']=='tracking'),{})
            history=[{'role':'user','content':'How can I track my order?'},
                     {'role':'assistant','content':tracking.get('answer','Check the store website for tracking information.')}]
        body={'message':message,'history':history}
        try:
            req=Request('http://127.0.0.1:8000/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=100) as response:result=json.load(response)
            results.append({'scenario':name,'message':message,'history':history,**result})
            print(name,result['status'],result['answer'],flush=True)
        except HTTPError as exc:
            detail=json.load(exc).get('detail','Request failed')
            results.append({'scenario':name,'message':message,'error':detail,'http_status':exc.code})
            print(name,exc.code,detail,flush=True)
        (reports/'generation_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('Recorded',len(results),'HTTP scenarios.',flush=True)
