
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image
import cv2

# 설정 및 모델 로드

MODEL_PATH = "checkpoints/best_densenet121.keras"
LAST_CONV_LAYER_NAME = "relu"

st.set_page_config(page_title="KNEE X-ray Classifier", layout="wide")

# 모델 로딩
@st.cache_resource
def load_keras_model():
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={'preprocess_input': preprocess_input}
        )
        return model
    except Exception as e:
        st.error(f"모델 로딩 중 오류가 발생했습니다: {e}")
        st.error(f"'{MODEL_PATH}' 경로에 모델 파일이 있는지 확인해주세요.")
        return None

model = load_keras_model()

# Grad-CAM 함수 구현

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    if model is None:
        return None

    base_model = model.get_layer('densenet121')
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    last_conv_layer_model = keras.Model(base_model.inputs, last_conv_layer.output)

    # 전체 모델의 최종 예측 출력을 얻기 위한 모델을 정의
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    for layer in model.layers:
        if layer.name not in ['input_image', 'data_augmentation', 'densenet_preprocess', 'densenet121']:
            x = layer(x)
    classifier_model = keras.Model(classifier_input, x)

    # GradientTape를 사용하여 그래디언트를 계산
    with tf.GradientTape() as tape:

        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)

        preds = classifier_model(last_conv_layer_output)
        class_output = preds[0]

    grads = tape.gradient(class_output, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def superimpose_gradcam(img, heatmap, alpha=0.4):
    if heatmap is None:
        return img

    # 히트맵을 원본 이미지와 같은 크기로 확대
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 히트맵을 0-255 범위의 8비트 이미지로 변환
    heatmap = np.uint8(255 * heatmap)

    # 'jet' 컬러맵 적용
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # 원본 이미지와 히트맵을 겹침
    superimposed_img = jet * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img

# Streamlit UI 구성

st.title("무릎 X-ray 관절염 분류 및 Grad-CAM 시각화")
st.write("DenseNet121 기반의 딥러닝 모델을 사용하여 무릎 관절염 진행 단계를 예측하고, AI가 어느 부분을 보고 판단했는지 히트맵으로 보여줍니다.")

if model is not None:
    uploaded_file = st.file_uploader("무릎 X-ray 이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(pil_img.resize((224, 224)))

        processed_img_for_pred = np.expand_dims(img_array.copy(), axis=0)

        processed_img_for_gradcam = preprocess_input(processed_img_for_pred.copy())


        if st.button("예측 실행"):
            with st.spinner('모델이 이미지를 분석 중입니다...'):
                prediction = model.predict(processed_img_for_pred)[0][0]

                if prediction >= 0.5:
                    class_label = "Negative"
                    probability = prediction * 100
                else:
                    class_label = "Positive"
                    probability = (1 - prediction) * 100

                result_text = f"**{class_label}**일 확률이 **{probability:.2f}%** 입니다."
                # ---------------------------------------------------------------------

                # Grad-CAM 생성
                heatmap = make_gradcam_heatmap(processed_img_for_gradcam, model, LAST_CONV_LAYER_NAME)

                # 원본 이미지 위에 히트맵 겹치기
                superimposed_img = superimpose_gradcam(img_array, heatmap)

                # 결과 출력
                st.subheader("분석 결과")
                if class_label == "Positive":
                    st.error(result_text)
                else:
                    st.success(result_text)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(pil_img, caption="원본 이미지", use_column_width=True)
                with col2:
                    st.image(superimposed_img, caption="Grad-CAM 분석 결과", use_column_width=True)

                st.info("""
                **Grad-CAM 해석:**
                - **붉은색 영역**은 모델이 'Positive'라고 판단하는 데 가장 큰 영향을 미친 부분입니다.
                - **푸른색 영역**은 판단에 거의 영향을 미치지 않은 부분입니다.
                - 이 시각화는 모델의 판단을 해석하는 데 도움을 주지만, 100% 정확한 의학적 진단을 의미하지는 않습니다.
                """)
else:
    st.warning("모델을 불러올 수 없습니다. 관리자에게 문의하세요.")
