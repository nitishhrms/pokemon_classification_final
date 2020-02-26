from flask import Flask,render_template,request
import label_it
app=Flask(__name__)

##m=joblib.load('model.pkl')

@app.route('/')
def Hello():
    return render_template('index.html')



@app.route('/',methods=['POST'])
def classification():
    if request.method=='POST':
        f=request.files['userfile']
        path="./static/{}".format(f.filename)
        f.save(path)
        image=path
        target1=label_it.ans(path)
        ##target1=str(target[0][0])
    return render_template('index.html',my_target=target1,my_image=image)

if __name__=='__main__':
    app.jinja_env.cache = {}
    app.run(debug=True)