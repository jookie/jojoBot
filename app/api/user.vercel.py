from http.server import BaseHTTPRequestHandler
from os.path import join
from posixpath import abspath, dirname
dir = dirname(abspath(__file__))
 
class handler(BaseHTTPRequestHandler):
 
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        
        # You can use the path relative to the project's base directory.
        # with open(join('data', 'file.txt'), 'r') as file:
        
        # Or you can use the path relative to the current file's directory.
        with open(join(dir, '..', 'data', 'file.txt'), 'r') as file:  
          for line in file:
            self.wfile.write(line.encode())
        return