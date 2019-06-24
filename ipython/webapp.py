from chatbot_part2 import response
from flask import Flask
from flask import render_template,request
# from flask_socketio import SocketIO

app= Flask(__name__)

@app.route('/')
def home():
    return render_template('chatbot_web.html')

@app.route('/chatting',methods=['POST','GET'])
def chatting():
    if request.method=="POST":
        msg=request.form['message']
        # reply=msg
        reply=msg
        reply_test=response(msg)
        app.logger.info("reply fetched is: ", reply_test)
        return render_template('chatbot_web.html', user_input=msg, bot_response=reply)
    elif request.method=="GET":
        msg=request.args.get('message')
        reply=response(msg)
        print(reply)
        return render_template('chatbot_web.html', user_input=msg, bot_response=reply)
    else:
        return "GET method used"

if __name__=='__main__':
    app.run(debug=True)