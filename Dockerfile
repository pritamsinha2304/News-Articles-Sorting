##!/bin/bash
FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
#ENTRYPOINT [ "./main_class.py" ]
#CMD [ "flask", "run", "-h", "0.0.0.0", "-p", "5000" ]
CMD [ "python", "main.py", "flask", "run", "-h", "0.0.0.0", "-p", "5000" ]