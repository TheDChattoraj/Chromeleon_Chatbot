from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Email, Regexp

# Allow only @thermofisher.com
THERMO_EMAIL_REGEX = r'^[A-Za-z0-9._%+-]+@thermofisher\.com$'

class EmailForm(FlaskForm):
    email = StringField("Work Email", validators=[
        DataRequired(),
        Email(message="Enter a valid email address"),
        Regexp(THERMO_EMAIL_REGEX, message="Only @thermofisher.com addresses allowed")
    ])
    submit = SubmitField("Send OTP")

class OTPForm(FlaskForm):
    otp = StringField("One-time code", validators=[
        DataRequired(),
        Regexp(r'^\d{6}$', message="Enter the 6-digit code sent to your email")
    ])
    submit = SubmitField("Verify")
