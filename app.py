from flask import Flask, request, render_template, send_from_directory
from google.cloud import storage
import os
import main  # Your existing Python code
import os

os.environ['GOOGLE_CLOUD_PROJECT'] = 'sentiment-analysis-379200'
storage_client = storage.Client()
app = Flask(__name__)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        folder = request.form.get('dropdown')
        urls = main.process_folder(folder)
        return render_template('display.html', urls=urls)  # Pass the URLs to the template
    else:
        # Get the list of files in the 'scores_magnitude' folder in the bucket
        bucket_name = 'sentiment-files'
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix='scores_magnitude/')
        files = set(blob.name.split('/')[1] for blob in blobs if '/' in blob.name)
        return render_template('upload.html', folders=files)  # Render the upload.html template
    
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)


