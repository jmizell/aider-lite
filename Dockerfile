# Use Ubuntu 22.04 lts as base image
FROM ubuntu:jammy-20240911.1

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3 and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-pexpect \
    python3-configargparse \
    python3-dotenv \
    python3-flake8 \
    python3-git \
    python3-jsonschema \
    python3-networkx \
    python3-numpy \
    python3-pathspec \
    python3-prompt-toolkit \
    python3-pygments \
    python3-pytest \
    python3-requests \
    python3-rich \
    python3-scipy \
    python3-yaml \
    python3-pyls \
    git \
    golang-1.23 \
    universal-ctags \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip3 install --no-cache-dir -U pip setuptools>=61.0.0

RUN ln -s /usr/lib/go-1.23/bin/go /usr/bin/go
RUN ln -s /usr/lib/go-1.23/bin/gofmt /usr/bin/gofmt
RUN go install golang.org/x/tools/gopls@latest \
    && mv /root/go/bin/gopls /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install the package
RUN pip3 install --no-deps .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["aider"]
