# Sign Language Recognition

A computer vision-based sign language recognition system that can recognize numbers from 1 to 10 using OpenCV and Keras.

## 👨‍💻 Team Members
- Sneha Chakraborty (21BAI10204)
- Ankit Choudary (21BAI10409)
- Aritra Ball (21BAI10398)
- Thanraman (21BAI10162)
- Kaviya Lakshmi (21BAI10314)

## 🔧 Technologies Used
- Python 3.7.4
- OpenCV 3.4.2
- Keras 2.3.1
- TensorFlow 2.0.0
- Jupyter Notebook / Google Colab

## 💡 Project Summary
This project detects hand gestures for numbers from 1 to 10 and is easily extendable to alphabets and custom gestures. The system uses a trained CNN model for gesture recognition.

## 🗂 Project Structure
- `create_gesture.py`: Script to collect gesture images
- `model_gesture.py`: Script to build and train the model
- `test_model.py`: Script to test real-time recognition

## 📁 Dataset
Custom dataset created via webcam input.

## ⚙️ How to Run
1. Clone the repo
2. Run `create_gesture.py` to collect gesture data
3. Train using `model_gesture.py`
4. Test live recognition with `test_model.py`

## 🔒 License
[MIT](LICENSE)
