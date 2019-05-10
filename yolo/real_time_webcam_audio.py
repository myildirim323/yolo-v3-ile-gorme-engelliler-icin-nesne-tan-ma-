# kütüphaneleri belirtir
import numpy as np
import time
import cv2
import os
import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment

# ses segmenti dönüştürücüsünün yolunu belirler
AudioSegment.converter = "C:/Users/MEHMET/Anaconda3/pkgs/ffmpeg-3.2.4-1/Library/bin/ffmpeg.exe"

# COCO sınıfı etiketlerini yükler (eğitilmiş YOLO modelimizin)
LABELS = open("yolo-coco/coco_tr.names").read().strip().split("\n")

# COCO veri setinde (80 sınıf) eğitilmiş YOLO ağırlık dosyasını yükler
print("Uygulama Yükleniyor...")
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")

# sadece YOLO'da ihtiyaç duyulan "çıkış" katmanını belirler
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Kamerayı açıp başlatır
cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []

while True:
	frame_count += 1
    # Kare kare(çerçeve) yakalama işlemini yapar
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	frames.append(frame)

	if frame_count == 300:
		break
	if ret:
		key = cv2.waitKey(1)
		if frame_count % 60 == 0:
			end = time.time()
			# çerçeve boyutlarını alıp ve bir blob(Binary Large Objects)'a yani ikili sayı sitemine dönüştürür
			(H, W) = frame.shape[:2]
			# Giriş görüntüsünden bir blob oluşturup sonraki frame ye geçer
			# YOLO nesne algılayıcısının geçişi, sınırlayıcı kutuları ve ilgili olasılıkları verir
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			# sırasıyla algılanan sınırlayıcı kutular, olasılıklar ve sınıf id lerinin listesini başlatır
			boxes = []
			confidences = []
			classIDs = []
			centers = []

			# katman çıktıları kadar döngü oluşturur
			for output in layerOutputs:
				# algılanan her çıkış kadar döngü oluşturur
				for detection in output:
					# Algılanan geçerli nesnenin sınıf id sini ve güvenirliğini (ör. olasılık) çıkarır
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# tespit edilen olasılığın minimum olasılıktan daha büyük olmasını 
					# sağlayarak zayıf tahminleri filtreler
					if confidence > 0.5:
						# Sınırlayıcı kutu koordinatlarını YOLO'daki gerçek değerinin
						# şekildeki görüntü boyutuna göre geriye doğru ölçeklendirmesi yapılır
						# Sınırlamanın merkez (x, y) koordinatlarını ve
                        # ardından kutuların enini, boyunu döndürür
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# sınırlayıcı kutunun üst ve sol köşesini çıkarmak için 
						# center (x, y) koordinatlarını kullanır
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# sınırlayıcı kutusunun koordinatları, olasılıklar ve sınıf idleri listesini günceller
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						centers.append((centerX, centerY))

			# Zayıf, üst üste gelen sınırlayıcı kutuları yok saymak için maksimum olmayanlar baskılanır
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

			texts = []

			# en az bir nesneyi tepsit ettiyse
			if len(idxs) > 0:
				# bulunan indeksler kadar döngü uygulanır
				for i in idxs.flatten():
					# nesnenin pozisyonları belirlenir
					centerX, centerY = centers[i][0], centers[i][1]
					
					# genişlik(yatay) belirlenir
					if centerX <= W/3: 
						W_pos = "solda "
					elif centerX <= (W/3 * 2):
						W_pos = "merkezde "
					else:
						W_pos = "sağda "
					
					# yükseklik(dikey) belirlenir
					if centerY <= H/3:
						H_pos = "üst "
					elif centerY <= (H/3 * 2):
						H_pos = "orta "
					else:
						H_pos = "alt "

					texts.append(H_pos + W_pos + LABELS[classIDs[i]] + ' ' + 'var ')

			print(texts)
			
			# seslendirme modülü(Google Text-to-Speech)
			if texts:
				description = ', '.join(texts) 
				tts = gTTS(description, lang='tr')
				tts.save('tts.mp3')
				tts = AudioSegment.from_mp3("tts.mp3")
				subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])

	cv2.imshow('Object Decetion', frame)
cap.release()
cv2.destroyAllWindows()
os.remove("tts.mp3")