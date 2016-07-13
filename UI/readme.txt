1)To run the UI locally, run:

cd Picture1000/UI/
python run.py

now direct your browser to http://localhost:5000

2)To set up a server, launch the EC2 and type

git clone https://github.com/davidlibland/Picture1000/
cd Picture1000/UI/
mod_wsgi-express start-server —-reload-on-changes picture1000.wsgi

now direct your browser to http://SERVER-IP:8000