from flask import Flask, render_template, request
from azure.storage.blob import BlobClient
import urllib.request
import json
import urllib.request
import json
import os
import ssl


app = Flask(__name__)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload():
   if request.method == 'POST':
      f = request.files['file']
      storage_account_key = "7MSa4yl/Re/t9IfDWi7m8fxKfAQdd/wPGUte+TAlKYvTbtQJgFkDG14JstsZh8diDNkHl/SHK+sEThrlTHobaQ=="
      storage_url = "https://capimages.blob.core.windows.net/"
      blob_client = BlobClient(storage_url, container_name="test", blob_name=f.filename, credential=storage_account_key)
      with open(f.filename, "rb") as data:
          blob_client.upload_blob(data)
      url = 'https://covidinference.northcentralus.inference.ml.azure.com/score'
      api_key = 'pblmcCaQvwWbgWIgp92fkr4y3aGc0qzH' # Replace this with the API key for the web service
      headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
      data = {"image":f.filename}
      body = str.encode(json.dumps(data))
      req = urllib.request.Request(url, body, headers)
      response = urllib.request.urlopen(req)
      result = response.read()
      return result

if __name__ == "__main__":
   app.run(debug=True)