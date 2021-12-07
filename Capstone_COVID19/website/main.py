from flask import Flask, render_template, request
from azure.storage.blob import BlobClient
import urllib.request
import json



app = Flask(__name__)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload():
   if request.method == 'POST':
      f = request.files['file']
      storage_account_key = "LSRjcStHbWkkpPEVhtWdLfqj/LnhSQ52JEfTIRuXaHPDRESWTFUiurvqW2YZl6McnXoCQp1HVt28YDg3XC7k3A=="
      storage_url = "https://mlstor.blob.core.windows.net/"
      blob_client = BlobClient(storage_url, container_name="images", blob_name=f.filename, credential=storage_account_key)
      with open(f.filename, "rb") as data:
          blob_client.upload_blob(data)
      url = 'https://covid-inference.northcentralus.inference.ml.azure.com/score'
      api_key = '2ja9yDjBX8de4h8ynWpTsfI7YkuLL8BF' # Replace this with the API key for the web service
      headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
      data = {"image":f.filename}
      body = str.encode(json.dumps(data))
      req = urllib.request.Request(url, body, headers)
      response = urllib.request.urlopen(req)
      result = response.read()
      return result

if __name__ == "__main__":
   app.run(debug=True)