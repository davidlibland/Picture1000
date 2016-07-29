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
import ast
import uuid
import threading

sess,p_sample,word_to_id,id_to_word,themes,args=s.load_model()
# Run the poetry generation algorithm.
classify_image.maybe_download_and_extract()

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
            # filename = secure_filename(file.filename)
            filename=str(uuid.uuid4())
            path_to_image=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # save the file
            file.save(path_to_image)

            
            # Create a thread to process the image and write the poem.
            threading.Thread(target=Add_Poem_and_Pic_to_DB,args=(path_to_image,filename)).start()
            
            return redirect(url_for('index'))
        if allowed_file(file.filename):
            return render_template('upload.html')
    return render_template('upload.html')

def Add_Poem_and_Pic_to_DB(path_to_image,filename):
    #image segmentation          
    segments=classify_image.run_inference_on_image(path_to_image)

    #emotion analysis
    with open(path_to_image,'rb') as f:
        image_data=f.read()
    emotions = json.dumps(EmoAPI.sentiment_analysis(image_data))

    #print('Emotions ',(emotions),type(emotions))
    #print('Segments ',(segments),type(segments))
    #if segments: 
    #    segments=ast.literal_eval('{"'+str(segments.replace(' (score =','": ').replace(')',',"'))[0:-1]+'}')
    if emotions:     
        emotions=ast.literal_eval('{'+emotions[emotions.find('scores')+10:-3]+'}')
    #print('Emotions ',(emotions),type(emotions))
    #print('Segments ',(segments),type(segments))
    
    cur_themes,cur_weights = s.clean_themes(themes,{**segments,**emotions})
    txt = s.multi_theme_sample(sess,p_sample,word_to_id,id_to_word,cur_themes,cur_weights,args)
    print(dict(zip(cur_themes,cur_weights)))

    db_keywords=Poem(filename=filename,poem_txt=txt,lastrating=0,meanrating=0,votes=0)

    #fnames = poem.query.all()  
    #for u in fnames:
    #    db.session.delete(fnames)          
    
    db.session.add(db_keywords)
    db.session.commit()
    
    
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
                               
#sess.close()