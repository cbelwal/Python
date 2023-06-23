import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def sendEmail(toEmail,subject):
    body = "Resume of Chaitanya Belwal is attached"
    fromEmail = "Chaitanya Belwal <support@tradocly.com>" #"cbelwal@gmail.com"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = fromEmail 
    message["To"] = toEmail
    message["Subject"] = subject
    
    SMTPServer = "mail.tradocly.com"
    SMTPServerUserID = "s****@tradocly.com"
    SMTPServerUserPwd = "****"
    # Add body to email
    message.attach(MIMEText(body, "plain"))

    filename = "resume.pdf"  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file Base64
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    print("Sending email to ",toEmail,"...")
    with smtplib.SMTP(SMTPServer, 8889) as server: #, context=context
        server.login(SMTPServerUserID, SMTPServerUserPwd)
        server.sendmail(fromEmail, toEmail, text)