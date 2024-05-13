# Skin Cancer Detection with Deep Learning
Skin cancer detection project for Unstructured Data Analysis subject from the Master in Big Data at Universidad Pontificia Comillas. The goal of this project is to develop a Deep Learning model that can classify images of skin lesions into benign or malignant. The dataset used is from [Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) and contains 1800 images of skin lesions.

## ⚕️ Dataset
The dataset used for this project is from [Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) and contains 3297 images of benign skin moles and malignant skin moles. The images are in JPG format and have a resolution of 224x224 pixels. The dataset is divided into two folders, one for each class and into training and testing sets (80% and 20% respectively).

A sample of the images is shown below:

<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <img src="https://storage.googleapis.com/kagglesdsdata/datasets/174469/505351/test/benign/1025.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240507%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240507T144627Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=3eebf506a89da814246a4110ceab8ed94d7b6c34e62bd2619e68532a9a4e9229c626e4a7905dee09171b49854f585b55f71b4b584fb2dd71389a78fd4f952033409cc861d292ebefd090d51c8108417b256f9daae592d864a93d23dd5ee987fda89d8373ea5d5c33ff93e4a0221639df43ec7a8ebcc07b5295399c33a718b9d48ecb502afa005d4942e506c1c49b90132303ca2da21baafbeef50f94ffd1c54c81ef0cdbd93451c70ca7512d36b2dd3fbcb9978f7e335d5a83d8cee323d7bff208dde6d57480d0cc62b8db69475f816ec966375819f7c8edccb566af7cd1bfff93eea245e7f4337891c7d29cc9ef643c5ac403bb4e2665d206344e2e8d255c65" alt="Benign mole" style="width: 80%;">
    <p>Benign</p>
  </div>

  <div style="text-align: center;">
    <img src="https://storage.googleapis.com/kagglesdsdata/datasets/174469/505351/test/malignant/1074.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240507%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240507T062123Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=4f9c08a181e418be61e87ec2e931aecc90c357cf1d2ae89f06711336287e40b0505dc0e98236e8eb696ba0b70835bb46a3499768cd754326c30af63291b6d0de8422b7cf3975d8dfd7475242b483abc2ccb77e58bee147f050c47094190494973edaf337f4327794511c1b980314bf2c8c2b1bb2fbf75d8c6b6156e5a87f4c23fd9b37d2d8d31ce4cd2833b7c6d9b6ebc64b812665915d7e8f6ab7dbd5dd5117ed76d1d9bd38ad9d0e94da19100574b647327079fb196376213f7e71fa41f1e54dd407c3b38cf2a8f34d6e4331a75739fd705ac93a3ec5d5edbc9ff1d2af9d53a13fd2d739870132064961770abc2c12144ec32f8e0b93876d1b74d0aff187f9" alt="Malignant mole" style="width: 80%;">
    <p>Malignant</p>
  </div>
</div>

## 🧠 Deep Learning Techniques
Several Deep Learning techniques have been used to classify the images, from more traditional models like Dense Neural Networks to more complex models like Transfer Learning with Convolutional Neural Networks.
- [ ] Dense Neural Network
- [ ] Convolutional Neural Network from scratch
- [ ] Convolutional Neural Network with Data Augmentation
- [ ] Transfer Learning with ResNet152
- [ ] Transfer Learning with ?

## 🛠️ Technologies
- Python
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Streamlit
- Pandas

## 🚀 Deployment
An interactive web application has been developed using Streamlit to classify images of skin lesions. The application allows the user to upload an image and get the prediction of the model. The application is hosted on Streamlit Sharing and can be accessed [here]().

## 👩🏻‍💻 Authors
- [Blanca Martínez Rubio](https://github.com/blancamartnez)
- [María Belén Salgado Bottaro](https://github.com/MARIABELENSB)
- [Elena Cabrera Casquet](https://github.com/elena-cabrera)
