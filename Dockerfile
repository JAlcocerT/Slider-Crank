# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
#source-destination

# Install production dependencies.
RUN pip install -r requirements.txt

EXPOSE 8050

CMD python ./app/app.py 

### BUILD and PUSH LOCAL IMAGE TO DOCKERHUB

#Building from local files:
#docker build -t slidercrank .
#docker image ls #to check the images locally

#Running the container:
#docker run -p 8050:8050 slidercrank
#docker run --name slidercrank -p 8050:8050 slidercrank -d

#Pushing the image to the registry:

##dockerhub
#docker build -t reisikei/slidercrank .
#docker login
#docker push reisikei/slidercrank