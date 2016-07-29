from app import db


class Poem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(128), index=True, unique=True)
    poem_txt = db.Column(db.String(400), index=True, unique=False)
    lastrating = db.Column(db.Integer, index=True, unique=False)
    meanrating = db.Column(db.Float, index=True, unique=False)
    votes = db.Column(db.Integer, index=True, unique=False)
    
    def get_id(self):
        try:
            return unicode(self.id)  # python 2
        except NameError:
            return str(self.id)  # python 3

    def __repr__(self):
        return '<Poem %r>' % (self.poem)