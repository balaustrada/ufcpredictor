FROM pytorch/pytorch:latest

# Install the repository directly from GitHub
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install git+https://github.com/balaustrada/ufcscraper.git
RUN pip install git+https://github.com/balaustrada/ufcpredictor.git

ADD app.py .
ADD data .

# Expose the port the app runs on (if needed)
EXPOSE 7860

# Command to run your Jupyter notebook (you can change this as necessary)
CMD ["python", "app.py", "--data-folder", ".", "--model-path", "model.pth", "--port", "7860"]