FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /code

# Install the repository directly from GitHub
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


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

# Install packaged from copied source
RUN pip install .

# Copy input data (if no HuggingFace token is provided)
#ADD --chown=user /path/to/data/folder $HOME/app/data

# Expose the port the app runs on (if needed)
EXPOSE 7860

# Make data directory
RUN mkdir -p $HOME/app/data

# Command to run your Jupyter notebook (you can change this as necessary)
CMD ["ufcpredictor_app", "--data-folder", "data", "--port", "7860", "--server-name", "0.0.0.0", "--download-dataset", "--config-path", "configurations/time_evolution.yaml"]
