from flask import render_template, flash, redirect, session, url_for, request, g, send_from_directory
from werkzeug.utils import secure_filename
import os
from app import app, db
from .models import Poem
from app import classify_image
from app import EmoAPI

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/processImage', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Check that the file name passed is safe...
            filename = secure_filename(file.filename)
            # save the file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Run the poetry generation algorithm.
            classify_image.maybe_download_and_extract()
            path_to_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(path_to_image) as f:
                image_data=f.read()
            sent_anal_result = EmoAPI.sentiment_analysis(image_data)
            #theme_txt=classify_image.run_inference_on_image(path_to_image)
            #poem_txt=theme_txt #gruChar.sample_rnn(seed=theme_txt,restore=True)
            
            poem=Poem(filename=filename,poem=sent_anal_result)
            db.session.add(poem)
            db.session.commit()
            return redirect(url_for('index'))
    return render_template('upload.html')   
    
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',img_poems=Poem.query.all())

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
                               
