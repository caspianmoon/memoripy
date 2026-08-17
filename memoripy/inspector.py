from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .client import MemoryClient
from .service import MemoryService


def inspector_html(*, title: str = "Memoripy Inspector", api_prefix: str = "/v4") -> str:
    config = json.dumps({"apiPrefix": api_prefix})
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin:0; background:#0b0d10; color:#f4f6f8; }}
header {{ padding:24px 32px; border-bottom:1px solid #2a3038; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }}
main {{ padding:24px 32px 64px; display:grid; grid-template-columns:minmax(300px,1fr) minmax(360px,1.3fr); gap:20px; }}
section {{ border:1px solid #2a3038; background:#12161b; border-radius:14px; padding:18px; }}
input, textarea, select, button {{ font:inherit; border-radius:8px; border:1px solid #39414b; padding:10px 12px; background:#0e1216; color:inherit; }}
button {{ cursor:pointer; background:#eef2f5; color:#101418; font-weight:700; }}
.row {{ display:flex; gap:10px; flex-wrap:wrap; }} .row > input {{ flex:1; min-width:180px; }}
#memories button {{ display:block; width:100%; text-align:left; margin:8px 0; background:#171d23; color:#f4f6f8; }}
pre {{ white-space:pre-wrap; word-break:break-word; max-height:560px; overflow:auto; background:#090c0f; border-radius:10px; padding:14px; }}
.badge {{ padding:4px 8px; border:1px solid #46515e; border-radius:999px; font-size:12px; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; padding:18px; }} header {{ padding:18px; }} }}
</style>
</head>
<body>
<header><strong>Memoripy Inspector</strong><span class="badge">evidence-first</span>
<input id="token" type="password" placeholder="Bearer token or local API key"><button onclick="saveToken()">Save token</button><span id="status"></span></header>
<main>
<div>
<section><h2>Recall</h2><div class="row"><input id="query" placeholder="What should the agent remember?"><button onclick="recall()">Search</button></div><div id="memories"></div></section>
<section><h2>Audit</h2><button onclick="audit()">Run memory audit</button><pre id="audit"></pre></section>
</div>
<div>
<section><h2>Memory receipt</h2><pre id="detail">Select a result to inspect evidence, history, trust, temporal validity, and retrieval receipts.</pre></section>
<section><h2>Correct or forget</h2><input id="memoryId" placeholder="memory id" style="width:100%;box-sizing:border-box"><textarea id="correction" placeholder="corrected value" style="width:100%;box-sizing:border-box;margin-top:10px"></textarea><div class="row" style="margin-top:10px"><button onclick="correctMemory()">Correct</button><button onclick="forgetMemory()">Forget</button></div></section>
</div>
</main>
<script>
const CONFIG={config};
const tokenEl=document.getElementById('token'); tokenEl.value=localStorage.getItem('memoripyToken')||'';
function saveToken(){{localStorage.setItem('memoripyToken',tokenEl.value);setStatus('saved')}}
function setStatus(value){{document.getElementById('status').textContent=value}}
async function api(path, options={{}}){{
  const headers={{'Content-Type':'application/json',...(options.headers||{{}})}};
  const token=tokenEl.value.trim(); if(token) headers.Authorization='Bearer '+token;
  const response=await fetch(CONFIG.apiPrefix+path,{{...options,headers}});
  const text=await response.text(); let payload; try{{payload=JSON.parse(text)}}catch{{payload={{raw:text}}}}
  if(!response.ok) throw new Error(payload.detail||payload.error||response.statusText); return payload;
}}
async function recall(){{try{{setStatus('searching');const q=document.getElementById('query').value;const result=await api('/recall',{{method:'POST',body:JSON.stringify({{query:q,limit:20,include_trace:true}})}});renderMemories(result.results||[]);setStatus((result.results||[]).length+' results')}}catch(e){{setStatus(e.message)}}}}
function renderMemories(items){{const root=document.getElementById('memories');root.innerHTML='';items.forEach(item=>{{const memory=item.memory||{{}};const button=document.createElement('button');button.textContent=(memory.kind||'memory')+' · '+(memory.summary||memory.value||memory.record_id);button.onclick=()=>explain(memory.record_id,item);root.appendChild(button)}})}}
async function explain(id,searchItem){{try{{document.getElementById('memoryId').value=id;const data=await api('/memories/'+encodeURIComponent(id)+'/explain');document.getElementById('detail').textContent=JSON.stringify({{search:searchItem,explanation:data}},null,2)}}catch(e){{setStatus(e.message)}}}}
async function audit(){{try{{const data=await api('/audit');document.getElementById('audit').textContent=JSON.stringify(data,null,2)}}catch(e){{setStatus(e.message)}}}}
async function correctMemory(){{try{{const id=document.getElementById('memoryId').value;const value=document.getElementById('correction').value;const data=await api('/memories/'+encodeURIComponent(id)+'/correct',{{method:'POST',body:JSON.stringify({{value,reason:'Corrected through Memoripy Inspector'}})}});document.getElementById('detail').textContent=JSON.stringify(data,null,2);setStatus('corrected')}}catch(e){{setStatus(e.message)}}}}
async function forgetMemory(){{try{{const id=document.getElementById('memoryId').value;const data=await api('/memories/'+encodeURIComponent(id),{{method:'DELETE'}});document.getElementById('detail').textContent=JSON.stringify(data,null,2);setStatus('forgotten')}}catch(e){{setStatus(e.message)}}}}
</script>
</body></html>'''


class InspectorService:
    def __init__(self, client: MemoryClient, *, api_key: str | None = None) -> None:
        self.memory_service = MemoryService(client=client, api_key=api_key)

    def handle(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any],
        query: dict[str, list[str]],
        headers: dict[str, str],
    ) -> tuple[int, str, str]:
        route = path.rstrip("/") or "/"
        if method == "GET" and route in ("/", "/inspector"):
            return HTTPStatus.OK, "text/html; charset=utf-8", inspector_html()
        status, response = self.memory_service.handle_request(
            method=method,
            path=path,
            payload=payload,
            query=query,
            headers=headers,
        )
        return status, "application/json; charset=utf-8", json.dumps(response, ensure_ascii=False, default=str)


def serve_inspector(
    client: MemoryClient,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    api_key: str | None = None,
) -> ThreadingHTTPServer:
    service = InspectorService(client, api_key=api_key)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None: self._dispatch("GET")
        def do_POST(self) -> None: self._dispatch("POST")
        def do_PATCH(self) -> None: self._dispatch("PATCH")
        def do_DELETE(self) -> None: self._dispatch("DELETE")
        def log_message(self, format: str, *args: Any) -> None: return

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                payload = {}
            status, content_type, body_text = service.handle(
                method=method,
                path=parsed.path,
                payload=payload,
                query=parse_qs(parsed.query),
                headers={key: value for key, value in self.headers.items()},
            )
            body = body_text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)
