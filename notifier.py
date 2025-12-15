# notifier.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from dotenv import load_dotenv

load_dotenv()
FROM_EMAIL = os.getenv("SMTP_USER")
PASSWORD = os.getenv("SMTP_PASSWORD")

def notify_on_risks_simple(to_emails, subject, message, attachment_path):
    """
    Sends an email with a PDF attachment to multiple recipients.
    """
    for email in to_emails:
        msg = MIMEMultipart()
        msg["From"] = FROM_EMAIL
        msg["To"] = email
        msg["Subject"] = subject

        # Add message
        msg.attach(MIMEText(message, "plain"))

        # Attach PDF
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
            msg.attach(part)

        # Send email
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(FROM_EMAIL, PASSWORD)
            server.sendmail(FROM_EMAIL, email, msg.as_string())
            server.quit()
            print(f"✅ Email sent to {email}")
        except Exception as e:
            print(f"❌ Error sending email to {email}: {e}")
