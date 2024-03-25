# Project Name : Melanoma Skin Cancer

![Alt text](https://raw.githubusercontent.com/raviatkumar/Melanoma-Skin-Cancer/master/Image/skin_care.jpg)

## User Guide

### Serving Deep Learning Model through API

#### Installation

To install the required packages for this project, use the following command after creating a virtual environment:

```bash
pip install -r requirements.txt
```

*Note: The model was trained on Google Colab with GPU support.*

#### About Project

This project aims to serve a deep learning model through an API using FastAPI.The Melanoma Skin Cancer Image dataset comprises 10,000 images, aimed at aiding in the development of accurate deep learning models for the classification of melanoma, a deadly form of skin cancer. With 9,600 images designated for training purposes and an additional 1,000 images reserved for model evaluation, the dataset presents a valuable resource for enhancing the early detection and treatment of melanoma, potentially saving numerous lives. Leveraging advanced machine learning techniques on this dataset can facilitate the creation of robust models capable of accurately identifying melanoma from skin images, thereby improving diagnostic accuracy and enabling timely intervention strategies.

#### How to Run the App

After installing the necessary packages, run the following command from the project root directory to start the app:

```bash
uvicorn app.main:app
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) from your browser to access Swagger. You can upload an image through the predict endpoint and receive a JSON response. Use the `--reload` argument to see immediate effects when changing code.

Alternative 
```bash
 Postman
```

#### Running Tests

To run the test cases, execute the following command from the project root directory:

```bash
pytest
```

#### How to Run the App with Docker

Ensure you are in the project root directory and Docker is running. Use the following command to create a Docker image:

```bash
docker build -t image-classifier-api .
```

Once the image is built successfully, run the container with the following commands:

```bash
docker run -p 8000:80 image-classifier-api
```

Visit http://127.0.0.1:8000/docs from your browser to access Swagger. You can upload an image through the predict endpoint and receive a JSON response.

#### Model Training and Performance

The model performs relatively well with an accuracy of 86%. However, it seems to perform better in identifying "benign" cases compared to "malignant" cases, as indicated by the higher precision and recall values for the "benign" class. Further analysis and fine-tuning of the model may be required to improve its performance, especially for the "malignant" class.
