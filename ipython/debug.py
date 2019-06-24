from chatbot_part2 import response

while(True):
    user_input=input("->")
    reply=response(user_input)
    print(reply)