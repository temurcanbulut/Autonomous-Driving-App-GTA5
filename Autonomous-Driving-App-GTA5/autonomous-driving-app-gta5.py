import cv2
import numpy as np
import mss
import time

# YOLO model ve ağırlıkları yükle
model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
katman_isimleri = model.getLayerNames()
cikis_katmanlari = [katman_isimleri[i - 1] for i in model.getUnconnectedOutLayers()]

# Etiketler
siniflar = []
with open("coco.names", "r") as f:
    siniflar = [satir.strip() for satir in f.readlines()]

# Renkler
RENKLER = {
    "insan": (0, 255, 255),  # Sarı
    "bisiklet": (42, 42, 165),  # Kahverengi
    "araba": (0, 255, 0),  # Yeşil
    "motosiklet": (0, 165, 255),  # Turuncu
    "otobus": (128, 128, 128),  # Gri
    "tren": (255, 0, 255),  # Pembe
    "kamyon": (107, 142, 35),  # Haki
    "trafik_isigi": (0, 0, 255),  # Kırmızı
    "yangin_muslugu": (255, 255, 255),  # Beyaz
    "kendi_arabam": (255, 0, 255),  # Mor
    "diger": (255, 255, 255)  # Beyaz
}

# Canlı varlıklar ve diğer araçlar
canli_varliklar = ["insan"]
kendi_arabam = "araba"
diger_araclar = ["bisiklet", "motosiklet", "otobus", "tren", "kamyon", "trafik_isigi", "yangin_muslugu"]

# Mesafe hesaplama fonksiyonu
def mesafe_hesapla(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Ekran yakalama
with mss.mss() as ekran:
    monitor = ekran.monitors[1]

    while True:
        ekran_goruntusu = ekran.grab(monitor)
        goruntu = np.array(ekran_goruntusu)
        yukseklik, genislik, kanallar = goruntu.shape

        # Görüntüyü BGR formatına dönüştür
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGRA2BGR)
        blob = cv2.dnn.blobFromImage(goruntu, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        model.setInput(blob)
        ciktilar = model.forward(cikis_katmanlari)

        sinif_idler = []
        guvenler = []
        kutular = []
        for cikti in ciktilar:
            for tespit in cikti:
                skorlar = tespit[5:]
                sinif_id = np.argmax(skorlar)
                guven = skorlar[sinif_id]
                if guven > 0.5:
                    merkez_x = int(tespit[0] * genislik)
                    merkez_y = int(tespit[1] * yukseklik)
                    w = int(tespit[2] * genislik)
                    h = int(tespit[3] * yukseklik)
                    x = int(merkez_x - w / 2)
                    y = int(merkez_y - h / 2)
                    kutular.append([x, y, w, h])
                    guvenler.append(float(guven))
                    sinif_idler.append(sinif_id)

        indeksler = cv2.dnn.NMSBoxes(kutular, guvenler, 0.5, 0.4)
        en_yakin_arac_mesafesi = float('inf')
        en_yakin_arac_koordinatlari = None
        for i in range(len(kutular)):
            if i in indeksler:
                x, y, w, h = kutular[i]
                if sinif_idler[i] < len(siniflar):
                    etiket = str(siniflar[sinif_idler[i]])
                else:
                    etiket = "diger"

                # Renk belirleme
                if etiket in canli_varliklar:
                    renk = RENKLER["insan"]
                elif etiket == kendi_arabam:
                    renk = RENKLER["kendi_arabam"]
                elif etiket in diger_araclar:
                    renk = RENKLER[etiket]
                else:
                    renk = RENKLER["diger"]

                # En yakın araç hesaplama
                if etiket in diger_araclar or etiket == kendi_arabam:
                    merkez_x = x + w // 2
                    merkez_y = y + h // 2
                    mesafe = mesafe_hesapla(genislik // 2, yukseklik, merkez_x, merkez_y)
                    if mesafe < en_yakin_arac_mesafesi:
                        en_yakin_arac_mesafesi = mesafe
                        en_yakin_arac_koordinatlari = (merkez_x, merkez_y)

                cv2.rectangle(goruntu, (x, y), (x + w, y + h), renk, 2)
                cv2.putText(goruntu, etiket, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2)

        # En yakın aracın mesafesini ekrana yazdırma
        if en_yakin_arac_koordinatlari is not None:
            arac_mesafesi_text = f"Mesafe: {en_yakin_arac_mesafesi / 10:.1f} m"
            cv2.line(goruntu, (genislik // 2, yukseklik), en_yakin_arac_koordinatlari, (0, 255, 255), 2)
            cv2.putText(goruntu, arac_mesafesi_text, (en_yakin_arac_koordinatlari[0], en_yakin_arac_koordinatlari[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # İşlenmiş görüntüyü göster
        cv2.imshow("GTA 5 Islenmis", goruntu)

        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 

cv2.destroyAllWindows()
