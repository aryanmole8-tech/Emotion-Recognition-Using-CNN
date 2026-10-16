import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="😊",
    layout="centered"
)


# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_emotion_model():
    return load_model(
        "final_emotion_recognition_model.keras"
    )


model = load_emotion_model()


# --------------------------------
# LOAD FACE DETECTOR
# --------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# --------------------------------
# EMOTION LABELS
# --------------------------------

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# --------------------------------
# TITLE
# --------------------------------

st.title("😊 Emotion Recognition System")

st.write(
    "Detect facial emotions using an uploaded image "
    "or live camera."
)


# --------------------------------
# SELECT MODE
# --------------------------------

mode = st.radio(
    "Select Mode",
    ["📁 Upload Image", "🎥 Live Camera"]
)


# ==================================================
# IMAGE UPLOAD MODE
# ==================================================

if mode == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        # Convert RGBA to RGB if necessary
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:

            image_np = cv2.cvtColor(
                image_np,
                cv2.COLOR_RGBA2RGB
            )

        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:

            st.warning("No face detected.")

        else:

            # Select largest face
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            face = gray[y:y+h, x:x+w]

            face_resized = cv2.resize(
                face,
                (48, 48)
            )

            face_normalized = (
                face_resized.astype("float32") / 255.0
            )

            face_input = face_normalized.reshape(
                1,
                48,
                48,
                1
            )

            prediction = model.predict(
                face_input,
                verbose=0
            )

            predicted_class = np.argmax(prediction)

            predicted_emotion = emotion_labels[
                predicted_class
            ]

            confidence = (
                prediction[0][predicted_class] * 100
            )

            cv2.rectangle(
                image_bgr,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            text = (
                f"{predicted_emotion} "
                f"({confidence:.1f}%)"
            )

            cv2.putText(
                image_bgr,
                text,
                (x, max(y-10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            result_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB
            )

            st.success(
                f"Predicted Emotion: {predicted_emotion}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

            st.image(
                result_rgb,
                caption="Emotion Detection Result",
                use_container_width=True
            )


# ==================================================
# LIVE CAMERA MODE
# ==================================================

else:

    st.subheader("🎥 Live Emotion Detection")

    st.write(
        "Click START and allow browser camera access."
    )


    class EmotionVideoProcessor(VideoProcessorBase):

        def recv(self, frame):

            # Convert video frame to image
            image = frame.to_ndarray(
                format="bgr24"
            )

            # Convert to grayscale
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            # Process detected faces
            for (x, y, w, h) in faces:

                # Crop face
                face = gray[y:y+h, x:x+w]

                # Resize
                face_resized = cv2.resize(
                    face,
                    (48, 48)
                )

                # Normalize
                face_normalized = (
                    face_resized.astype("float32")
                    / 255.0
                )

                # Reshape
                face_input = face_normalized.reshape(
                    1,
                    48,
                    48,
                    1
                )

                # Predict
                prediction = model.predict(
                    face_input,
                    verbose=0
                )

                predicted_class = np.argmax(
                    prediction
                )

                predicted_emotion = emotion_labels[
                    predicted_class
                ]

                confidence = (
                    prediction[0][predicted_class]
                    * 100
                )

                # Draw face rectangle
                cv2.rectangle(
                    image,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                # Display emotion
                text = (
                    f"{predicted_emotion} "
                    f"{confidence:.1f}%"
                )

                cv2.putText(
                    image,
                    text,
                    (x, max(y-10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Return processed frame
            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )


    webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )