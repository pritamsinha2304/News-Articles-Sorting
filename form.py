from flask_wtf import FlaskForm
import wtforms as wtf
import wtforms.validators as wtval
from flask_wtf.file import FileAllowed


class FileExtensionChecking(object):
    def __init__(self):
        pass

    def __call__(self, form, field):
        if not field.data.filename.endswith('.csv'):
            raise wtval.ValidationError(f'The file you uploaded {field.data.filename} does not have .csv extension')


class SingleDataUploadForm(FlaskForm):
    s_input = wtf.TextAreaField('Put some news article', validators=[wtval.InputRequired(), wtval.DataRequired()])

    s_model = wtf.SelectField('Select the model to predict', validators=[wtval.InputRequired(), wtval.DataRequired()],
                              choices=[('', '.....'), ('custom', 'Keras Custom Model')])
    s_opt = wtf.SelectField('Select the label', validators=[wtval.Optional()], choices=[('', '.....'),
                                                                                        ('entertainment', 'Entertainment'),
                                                                                        ('wellness', 'Wellness'),
                                                                                        ('politics', 'Politics'),
                                                                                        ('travel', 'Travel'),
                                                                                        ('style & beauty', 'Style & Beauty'),
                                                                                        ('parenting', 'Parenting'),
                                                                                        ('healthy & living', 'Healthy Living'),
                                                                                        ('business', 'Business'),
                                                                                        ('food & drink', 'Food & Drink'),
                                                                                        ('sports', 'Sports'),
                                                                                        ('comedy', 'Comedy'),
                                                                                        ('home & living', 'Home & Living'),
                                                                                        ('weddings', 'Weddings'),
                                                                                        ('impact', 'Impact'),
                                                                                        ('divorce', 'Divorce'),
                                                                                        ('crime', 'Crime'),
                                                                                        ('media', 'Media'),
                                                                                        ('religion', 'Religion'),
                                                                                        ('arts & culture', 'Arts & Culture'),
                                                                                        ('tech', 'Tech'),
                                                                                        ('style', 'Style'),
                                                                                        ('science', 'Science'),
                                                                                        ('world_news', 'World News'),
                                                                                        ('money', 'Money'),
                                                                                        ('arts', 'Arts'),
                                                                                        ('environment', 'Environment'),
                                                                                        ('college', 'College'),
                                                                                        ('education', 'Education')
                                                                                        ])
    s_upload = wtf.SubmitField('Upload and Predict')


class MultipleDataUploadForm(FlaskForm):
    m_input = wtf.FileField('Upload File',
                            validators=[wtval.InputRequired(), wtval.DataRequired()])

    m_model = wtf.SelectField('Select the model to predict', validators=[wtval.InputRequired(), wtval.DataRequired()],
                              choices=[('', '.....'), ('custom', 'Keras Custom Model')])
    m_opt = wtf.FileField('Upload actual label', validators=[wtval.Optional()])
    m_upload = wtf.SubmitField('Upload and Predict')


class DemoForm(FlaskForm):
    d_model = wtf.SelectField('Select which model to predict', validators=[wtval.InputRequired(), wtval.DataRequired()],
                              choices=[('', '.....'), ('custom', 'Keras Custom Model')])
    d_upload = wtf.SubmitField('Upload and Predict')


class SingleMultipleChoiceForm(FlaskForm):
    sm_choice = wtf.RadioField('Single Data', validators=[wtval.InputRequired(), wtval.DataRequired()],
                               choices=[('single', 'Single Data'), ('multiple', 'Multiple Data'), ('demo', 'Demo Data')])


