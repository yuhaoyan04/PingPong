"""Optional local Chrome/CDP smoke test for the v11 review (websocket-client)."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request
from urllib.parse import urlparse,unquote
from html.parser import HTMLParser

import websocket


class References(HTMLParser):
    def __init__(self):super().__init__();self.files=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in {'href','src','poster'} and value and not urlparse(value).scheme and not value.startswith('#'):
                self.files.append(unquote(value.split('#')[0]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('output/system_iteration_v11'))
    root=parser.parse_args().output.resolve();refs=References();refs.feed((root/'review.html').read_text())
    for path in refs.files:assert (root/path).is_file(),path
    os.environ['no_proxy']='127.0.0.1,localhost'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='pingpong-v11-browser-') as tmp:
        log=(root/'chrome_validation.log').open('w')
        process=subprocess.Popen(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
            '--disable-background-networking','--no-first-run',f'--user-data-dir={tmp}/profile',
            f'--remote-debugging-port={port}',(root/'review.html').as_uri()],stdout=log,stderr=log)
        ws=None
        try:
            deadline=time.monotonic()+25
            while True:
                try:
                    tabs=json.load(opener.open(f'http://127.0.0.1:{port}/json',timeout=1))
                    tab=next(t for t in tabs if t.get('type')=='page');break
                except Exception:
                    if time.monotonic()>deadline:raise
                    time.sleep(.1)
            ws=websocket.create_connection(tab['webSocketDebuggerUrl'],timeout=10,suppress_origin=True,http_no_proxy=['127.0.0.1','localhost'])
            token=0;errors=[]
            def call(method,params=None):
                nonlocal token
                token+=1;current=token;ws.send(json.dumps({'id':current,'method':method,'params':params or {}}))
                while True:
                    message=json.loads(ws.recv())
                    if message.get('method')=='Runtime.exceptionThrown':errors.append(message['params']['exceptionDetails'])
                    if message.get('id')==current:
                        if 'error' in message:raise RuntimeError(message['error'])
                        return message.get('result',{})
            def evaluate(expression):
                result=call('Runtime.evaluate',{'expression':expression,'returnByValue':True})
                if result.get('exceptionDetails'):raise RuntimeError(result['exceptionDetails'])
                return result['result'].get('value')
            def until(expression):
                deadline=time.monotonic()+20
                while not evaluate(expression):
                    if time.monotonic()>deadline:raise TimeoutError(expression)
                    time.sleep(.1)
            call('Runtime.enable');call('Page.enable')
            call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
            until("document.querySelectorAll('video').length===7")
            evaluate("document.querySelector('video').load();true")
            until("document.querySelector('video').readyState>=1")
            evaluate("document.querySelector('video').currentTime=40.1/25;true")
            until("!document.querySelector('video').seeking && document.querySelector('video').readyState>=2")
            evaluate("const v=document.querySelector('video');v.nextElementSibling.querySelectorAll('button')[1].click();true")
            until("!document.querySelector('video').seeking && Math.abs(document.querySelector('video').currentTime-41.1/25)<.001")
            evaluate("document.querySelector('video').nextElementSibling.querySelectorAll('button')[0].click();true")
            until("!document.querySelector('video').seeking && Math.abs(document.querySelector('video').currentTime-40.1/25)<.001")
            assert evaluate("document.querySelector('video').videoWidth===1280 && document.querySelector('video').videoHeight===1000")
            count=evaluate("document.querySelectorAll('[data-review]').length")
            expected=len(json.loads((root/'review_queue.json').read_text()));assert count==expected
            evaluate("const s=document.querySelector('[data-review]');s.value='uncertain';s.dispatchEvent(new Event('change'));true")
            call('Page.reload');until("document.querySelector('[data-review]')?.value==='uncertain'")
            download=Path(tmp)/'downloads';download.mkdir()
            call('Browser.setDownloadBehavior',{'behavior':'allow','downloadPath':str(download)})
            evaluate("document.getElementById('export').click();true")
            deadline=time.monotonic()+20;path=download/'pingpong_v11_human_review.json'
            while not path.exists():
                if time.monotonic()>deadline:raise TimeoutError('review JSON download')
                time.sleep(.1)
            result=json.loads(path.read_text());assert len(result['items'])==count
            assert result['items'][0]['review_status']=='uncertain'
            assert all(r['suggestion_is_not_a_label'] for r in result['items'])
            assert not errors,errors
            report={'passed':True,'browser':'Chrome headless/CDP','relative_resources_checked':len(refs.files),
                    'videos':7,'next_previous_frame_tested':True,'review_queue_items':count,
                    'review_local_storage_tested':True,'review_json_export_tested':True,
                    'source_labels_modified':False,'runtime_exceptions':errors}
            (root/'browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(json.dumps(report,ensure_ascii=False,indent=2))
        finally:
            if ws:ws.close()
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()


if __name__=='__main__':main()
