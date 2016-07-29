from flask import render_template, flash, redirect, session, url_for, request, g, send_from_directory
from werkzeug.utils import secure_filename
import os
from app import app, db
from .models import Poem
from app import classify_image
from app import EmoAPI
from sqlalchemy import desc
import json
import Poet.sample as s
import numpy as np

sess,p_sample,word_to_id,id_to_word,themes,args=s.load_model()

import sys

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
        # submit an empty part without filename
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

            #image segmentation          
            segments=classify_image.run_inference_on_image(path_to_image)

            #emotion analysis
            with open(path_to_image,'rb') as f:
                image_data=f.read()
            emotions = json.dumps(EmoAPI.sentiment_analysis(image_data))

            print('Emotions ',(emotions),type(emotions))
            print('Segments ',(segments),type(segments))
            
            sample_theme=np.random.choice(list(themes.keys()))
            sample_theme_ID=word_to_id[sample_theme]
            txt=s.sample_from_active_sess_with_theme(sess,p_sample,word_to_id,id_to_word,sample_theme,args).replace(' <eop>','')

            db_keywords=Poem(filename=filename,segments=txt,emotions=emotions,lastrating=0,meanrating=0,votes=0)

            #fnames = poem.query.all()  
            #for u in fnames:
            #    db.session.delete(fnames)          
            
            db.session.add(db_keywords)
            db.session.commit()
            return redirect(url_for('index'))
        if allowed_file(file.filename):
            return render_template('upload.html')
    return render_template('upload.html')
    
@app.route('/')
@app.route('/index')
def index():
    db=Poem.query.order_by(desc(Poem.meanrating)).all()# order_by(Poem.meanrating)
    return render_template('index.html',img_poems=db)


@app.route('/about')
def about():
    return render_template('about.html',title='About')

@app.route('/rate',methods=['POST'])
def rate():
    img_poems=Poem.query.all()
    
    for img_poem in img_poems:
        stars=request.form['stars'+str(img_poem.id)]
        xn=img_poem.lastrating
        img_poem.lastrating=stars
        db.session.commit() 
        if img_poem.lastrating == 0:
           img_poem.lastrating=xn
        else:
            mn_n=img_poem.meanrating
            vts_n=img_poem.votes
            img_poem.votes=img_poem.votes+1
            img_poem.meanrating=(mn_n*vts_n+img_poem.lastrating)/img_poem.votes
        db.session.commit()
    return redirect('/')

@app.route('/rating')
def rating():
    return render_template('rating.html',img_poems=Poem.query.all())

@app.route('/uploads/<filename>') 
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)