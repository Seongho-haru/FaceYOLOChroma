import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.uic import loadUi
from facenet_pytorch import InceptionResnetV1
import chromadb
import torch
import numpy as np
from ultralytics import YOLO
import os

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load UI file
        loadUi('main.ui', self)

        # OpenCV VideoCapture
        self.cap = cv2.VideoCapture(0)  # 0번 카메라 사용 (기본 카메라)

        # QTimer 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms마다 업데이트 (약 33 FPS)

        # 이벤트 연결
        self.register_edit.textChanged.connect(self.isRegisterEdit)
        self.start_btn.clicked.connect(self.StartPredict)
        self.stop_btn.clicked.connect(self.StopPredict)
        self.register_btn.clicked.connect(self.RegisterFace)

        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        self.model = YOLO('face_m.pt')  # YOLOv8 얼굴 검출 모델 가정

        # ChromaDB 초기화
        self.chroma_client = chromadb.PersistentClient()
        self.collection = self.chroma_client.get_or_create_collection(name="face_embeddings")

        self.isPredict = False

        # 얼굴 저장 디렉토리 생성
        os.makedirs("saved_faces", exist_ok=True)

    def update_frame(self):
        # 카메라 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            return

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.isPredict:
            # YOLO 추론 (person 클래스만 활성화)
            results = self.model.predict(rgb_frame, classes=[0])  # 0번 클래스는 "person"

            for box in results[0].boxes:
                xyxy = box.xyxy[0].int().tolist()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf[0])  # 신뢰도

                # 얼굴 영역 추출
                face_image = rgb_frame[y1:y2, x1:x2]

                # 임베딩 생성
                embedding = self.get_embedding_from_face(face_image)
                if embedding is None:
                    label_text = "Unknown"
                else:
                    # ChromaDB 검색
                    query_results = self.collection.query(
                        query_embeddings=[embedding],
                        n_results=1,
                    )
                    print(query_results)

                    if len(query_results["documents"]) > 0:
                        dist = query_results["distances"][0][0]
                        if dist < 0.2:  # 유사도 임계값
                            label_text = query_results["documents"][0][0]
                        else:
                            label_text = "Unknown"
                    else:
                        label_text = "Unknown"

                # 라벨 및 바운딩 박스 표시
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_frame, f"{label_text}_{conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 화면 표시
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.camera.setPixmap(QPixmap.fromImage(qimg))


    def isRegisterEdit(self):
        if self.register_edit.text():
            self.register_btn.setEnabled(True)
        else:
            self.register_btn.setEnabled(False)

    def StartPredict(self):
        self.isPredict = True

    def StopPredict(self):
        self.isPredict = False

    def RegisterFace(self):
        # 얼굴을 등록 (현재 프레임 캡처 후 YOLO로 얼굴 검출 → 임베딩)
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.predict(rgb_frame, classes=[0])

            if not results or not results[0].boxes:
                QMessageBox.warning(self, "Warning", "No face detected!")
                return

            # 첫 번째 얼굴만 등록
            box = results[0].boxes[0]  # 첫 번째 박스 가져오기
            box = box.xyxy[0].tolist()  # 리스트로 변환
            x1, y1, x2, y2 = map(int, box)  # 각 값을 int로 변환
            face_image = rgb_frame[y1:y2, x1:x2]
            user_name = self.register_edit.text().strip()
            cv2.imwrite(f"saved_faces/{user_name}.jpg", face_image)

            # 임베딩 추출
            embedding = self.get_embedding_from_face(face_image)
            if embedding is None:
                QMessageBox.warning(self, "Warning", "Failed to generate embedding!")
                return

            user_id = f"user_{user_name}"  # 고유 ID 생성
            self.collection.add(
                ids=[user_id],  # 고유 ID
                documents=[user_name],
                embeddings=[embedding],
                metadatas=[{"source": "camera"}]
            )

            # 이미지도 저장
            save_path = os.path.join("saved_faces", f"{user_name}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Success", f"Face registered for {user_name}")


    def get_embedding_from_face(self, face_image):
        # face_image는 RGB, 임의 크기
        if face_image.size == 0:
            return None

        # FaceNet용 전처리
        face_image = cv2.resize(face_image, (160, 160))
        face_image = np.float32(face_image)
        face_image = (face_image - 127.5) / 128.0
        face_tensor = torch.from_numpy(face_image).permute(2,0,1).unsqueeze(0)

        with torch.no_grad():
            embedding = self.facenet_model(face_tensor).numpy()
        return embedding.flatten().tolist()

    def closeEvent(self, event):
        # 프로그램 종료 시 카메라 해제
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
