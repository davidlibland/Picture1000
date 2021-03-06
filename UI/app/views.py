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
from collections import OrderedDict
from PIL import Image



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
            #filename = secure_filename(file.filename)
            # get the extension
            #_,file_extension = os.path.splitext(filename)
            base_filename=str(uuid.uuid4())
            filename = base_filename + '.jpg'
            path_to_image=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # save the file
            file.save(path_to_image)
            with Image.open(file) as im:
                #im.save(path_to_image,'JPEG')
                size = 200, 128
                im.thumbnail(size, Image.ANTIALIAS)
                thumbnail_filename = base_filename + "_thumbnail.jpg"
                im.save(os.path.join(app.config['UPLOAD_FOLDER'], thumbnail_filename),'JPEG',optimize=True,quality=95)

            
            edit=0
            
            # Create a thread to process the image and write the poem.
            threading.Thread(target=Add_Poem_and_Pic_to_DB,args=(path_to_image,filename,thumbnail_filename,edit)).start()
            
            return redirect(url_for('index'))
        if allowed_file(file.filename):
            return render_template('upload.html')
        else:
            flash('Currently, we only accept jpeg files.')
    return render_template('upload.html')

def Add_Poem_and_Pic_to_DB(path_to_image,filename,thumbnail_filename,edit):

    #image segmentation          
    segments=classify_image.run_inference_on_image(path_to_image)

    #emotion analysis
    with open(path_to_image,'rb') as f:
        image_data=f.read()
    emotions = json.dumps(EmoAPI.sentiment_analysis(image_data))
        
    if emotions:   
        if emotions.find('face') < emotions.find('scores'):
            emotions=ast.literal_eval('{'+emotions[emotions.find('scores')+10:-3]+'}')
        else:
            emotions=ast.literal_eval('{'+emotions[emotions.find('scores')+10:emotions.find('face')-4]+'}')

    segm_dict = {**emotions,**segments}
    segm_list = sorted([(key,val) for key, val in segm_dict.items()],key = lambda x: -x[1])
    segm_txt=''.join('%s: %4.2f, ' % (key,val) for key, val in segm_list)
    
    cur_themes,cur_weights = s.clean_themes(themes,{**segments,**emotions})
    txt = s.multi_theme_sample(sess,p_sample,word_to_id,id_to_word,cur_themes,cur_weights,args)
    print('Themes ',dict(zip(cur_themes,cur_weights)))
    
    theme_list = sorted([(key,val) for key, val in dict(zip(cur_themes,cur_weights)).items()],key = lambda x: -x[1])
    theme_txt=''.join('%s: %4.2f, ' % (key,val) for key, val in theme_list)

    db_keywords=Poem(filename=filename,thumbnail_filename=thumbnail_filename,
                segm_txt=segm_txt,theme_txt=theme_txt,poem_txt=txt,lastrating=0,meanrating=0,votes=0)
    
    if edit==0:
        db.session.add(db_keywords)
    else:
        tdb=db_keywords.query.filter_by(filename=filename).first()
        tdb.segm_txt=segm_txt
        tdb.theme_txt=theme_txt
        tdb.poem_txt=txt
    db.session.commit()
    
@app.route('/')
@app.route('/index')
def index():
    db=Poem.query.order_by(desc(Poem.meanrating)).all()# order_by(Poem.meanrating)
    return render_template('index.html',img_poems=db)


@app.route('/about')
def about():
    return render_template('about.html',title='About')

@app.route('/ingredients',methods=['POST','GET'])
@app.route('/ingredients/<int:id>',methods=['POST','GET'])
def ingredients(id=None):
    if request.method == 'POST':
        db=Poem.query.filter_by(id=id).first()# order_by(Poem.meanrating)
        filename=db.filename
        thumbnail_filename=db.thumbnail_filename
        path_to_image=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Create a thread to process the image and write the poem.
        edit=1
        threading.Thread(target=Add_Poem_and_Pic_to_DB,args=(path_to_image,filename,thumbnail_filename,edit)).start()
    else:
        if id is None: 
            db=Poem.query.order_by(desc(Poem.meanrating)).first()# order_by(Poem.meanrating)
        else: 
            db=Poem.query.filter_by(id=id).first()# order_by(Poem.meanrating)  
    return render_template('ingredients.html',img_poem=db)

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
def uploaded_thumbnail_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
                               
#sess.close()