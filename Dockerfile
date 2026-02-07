FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    USER=dtagnn \
    HOME=/home/dtagnn

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (mandatory for HF Spaces) with UID 1000
RUN useradd -m -u 1000 $USER

# Set working directory
WORKDIR $HOME/app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY assets/ ./assets/
COPY main.py ./main.py

# Install dependencies
# Note: we install with [molecule-gnn] extras for full functionality
RUN pip install --no-cache-dir -e ".[molecule-gnn]"

# Set up directories and permissions
RUN mkdir -p "$HOME/app/temp" "$HOME/app/runs" "$HOME/app/chembl_dbs" \
    && chown -R $USER:$USER $HOME

# Environment variables for the app
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    GRADIO_TEMP_DIR="$HOME/app/temp/"

# Switch to the non-root user
USER $USER

# Expose the Gradio port
EXPOSE 7860

# CMD to launch the app
CMD ["dta_gnn", "ui", "--host", "0.0.0.0", "--port", "7860"]
