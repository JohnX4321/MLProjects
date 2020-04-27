import http.server
import socketserver
from chatbot_search.search import chatbot_query

PORT=8002
DIRECTORY='public'


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=DIRECTORY,**kwargs)

    def do_POST(self):
        self.send_response(200)
        content_length=int(self.headers['Content-Length'])
        post_body=self.rfile.read(content_length)
        self.end_headers()
        print('User Query',post_body)
        chatbot_reply=chatbot_query(post_body)
        self.wfile.write(str.encode(chatbot_reply))


with socketserver.TCPServer(('',PORT),Handler) as httpd:
    print('Listening on port: ',PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()