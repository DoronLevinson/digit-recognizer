# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

COPY requirements.txt ./
# Install PyTorch CPU-only version first
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files into the container
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]



# # Use official lightweight Python image
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Copy all files into the container
# COPY . .

# # Install CPU-only PyTorch and TorchVision first
# RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu \
#     -f https://download.pytorch.org/whl/torch_stable.html

# # Install the rest of the dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose port for Streamlit
# EXPOSE 8501

# # Run the Streamlit app
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]