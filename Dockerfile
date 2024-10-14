FROM pytorch/pytorch:latest

WORKDIR /code

# Install the repository directly from GitHub
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install git+https://github.com/balaustrada/ufcscraper.git
RUN pip install git+https://github.com/balaustrada/ufcpredictor.git


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app
# setting the owner to the user
COPY --chown=user . $HOME/app

# Expose the port the app runs on (if needed)
EXPOSE 7860

# Command to run your Jupyter notebook (you can change this as necessary)
CMD ["python", "app.py", "--data-folder", "data", "--model-path", "data/model.pth", "--port", "7860", "--server-name", "0.0.0.0"]