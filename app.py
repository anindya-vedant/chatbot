##from chatbot_part2 import response
from chatbot_git import response
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
        # msg=request.form['message']
        # msg=str(msg)
        # # reply=msg
        # reply=msg
        # reply_test=response(msg)
        # app.logger.info("reply fetched is: ", reply_test)
        user_response=request.form['message']
        user_response = user_response.lower()
        reply=response(user_response)
        # if (user_response != 'bye'):
        #     if (user_response == 'thanks' or user_response == 'thank you'):
        #         flag = False
        #         print("ROBO: You are welcome..")
        #     else:
        #         if (greeting(user_response) != None):
        #             print("ROBO: " + greeting(user_response))
        #         else:
        #             print("ROBO: ", end="")
        #             print(response(user_response))
        #             sent_tokens.remove(user_response)
        return render_template('chatbot_web.html', user_input=user_response, bot_response=reply)
    elif request.method=="GET":
        msg=request.args.get('message')
        reply=response(msg)
        print(reply)
        return render_template('chatbot_web.html', user_input=msg, bot_response=reply)
    else:
        return "GET method used"

if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True)
