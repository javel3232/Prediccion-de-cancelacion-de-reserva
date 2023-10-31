# Use the official Python image as the base image
FROM python:3.10.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any required dependencies
RUN pip install -r requirements.txt

# Expose port in case of need
EXPOSE 8080

#IF YOU WANT THE DOCKER IMAGE TO EXECUTE BOTH SCRIPTS AUTOMATICALLY
# Start a Bash shell and execute commands automatically by removing the comment to the next line
#CMD ["/bin/bash", "-c", "python /app/scripts/train.py && python /app/scripts/predict.py"]

# Set the default command to run /bin/bash
CMD ["/bin/bash"]


#You can execute either the train.py by calling (python scripts/train.py)
#or predict.py by calling python scripts/predict.py