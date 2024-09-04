from flask import Flask, request, render_template
import SendEmail

app = Flask(__name__)

# set FLASK_APP=Form.py
# flask run 
# flask --app Form --debug run
@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    email = text
    send_email(email)
    response = "<b>Email has been sent to:</b>" + email
    return response

def send_email(emailAddress):
    SendEmail.sendEmail(emailAddress,"Resume of Chait Belwal")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    #SendEmail.sendEmail("cbelwal@gmail.com","Test Email")