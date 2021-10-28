from flask import Flask

app = Flask(__name__)

@app.route('/') # decorator
def home():
    '''viewing function - home page'''
    return "Welcome home! :)"; 


@app.route('/educative')
def leanr():
    '''viewing function - appended page'''
    return "Ready to learn!"

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 3001)
