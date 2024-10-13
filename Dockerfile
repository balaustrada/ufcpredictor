# Use a base image with Python
FROM python:3.12

# Install the repository directly from GitHub
RUN pip install git+https://github.com/balaustrada/ufcscraper.git
RUN pip install git+https://github.com/balaustrada/ufcpredictor.git

# Set the working directory
WORKDIR /data

# Copy the requirements.txt file into the container
#COPY app/requirements.txt .

# Install necessary Python packages
# RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of your application files into the container
# COPY app/ .

# Expose the port the app runs on (if needed)
EXPOSE 7860

# Command to run your Jupyter notebook (you can change this as necessary)
CMD ["python", "app.py", "--data-folder", ".", "--model-path", "model.pth", "--port", "7860"]