# main.py
"""
Sistema de Detección de Cascos de Seguridad con YOLOv8
SENATI - Ingeniería Civil
CORREGIDO: Detecta cuando NO hay casco visible
"""

import cv2
import numpy as np
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Importar configuración
try:
    from config import *
except ImportError:
    print("❌ Error: config.py no encontrado")
    sys.exit(1)

# ==================== LOGGING ====================
def setup_logging():
    logger = logging.getLogger('HelmetDetection')
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
        backupCount=LOG_BACKUP_COUNT, encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()


class HelmetDetectionSystem:
    """Sistema de detección de cascos"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Iniciando Sistema de Detección de Cascos")
        logger.info("=" * 60)
        
        errors, warnings = validate_config()
        for w in warnings:
            logger.warning(w)
        if errors:
            for e in errors:
                logger.error(e)
            if not MODEL_PATH.exists():
                logger.critical("Modelo no encontrado. Abortando.")
                sys.exit(1)
        
        self.model = self._load_model()
        self.audio = self._init_audio() if ENABLE_AUDIO else None
        self.twilio = self._init_twilio() if ENABLE_WHATSAPP else None
        
        # Estado
        self.frames_no_helmet = 0
        self.last_alert = None
        self.total_detections = 0
        self.alerts_sent = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.fps_time = time.time()
        self.fps_frames = 0
        
        # Detector de personas (para saber si hay alguien sin casco)
        self.person_detector = None
        self._init_person_detector()
        
        logger.info("✅ Sistema listo")
        logger.info("")
        logger.info("⚠️  LÓGICA DE DETECCIÓN:")
        logger.info("   - VERDE: Persona CON casco detectado")
        logger.info("   - ROJO:  Persona visible SIN casco")
        logger.info("")
    
    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info(f"Cargando modelo: {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
            logger.info(f"✅ Modelo cargado - Clases: {list(model.names.values())}")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            sys.exit(1)
    
    def _init_person_detector(self):
        """Inicializa detector de personas con Haar Cascade"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("✅ Detector de rostros inicializado")
        except Exception as e:
            logger.warning(f"Detector de rostros no disponible: {e}")
            self.face_cascade = None
    
    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.init()
            
            if not ALERT_SOUND_PATH.exists():
                self._create_alert_sound()
            
            sound = pygame.mixer.Sound(str(ALERT_SOUND_PATH))
            sound.set_volume(ALERT_VOLUME)
            logger.info("✅ Audio inicializado")
            return sound
        except Exception as e:
            logger.warning(f"Audio no disponible: {e}")
            return None
    
    def _create_alert_sound(self):
        try:
            from scipy.io import wavfile
            sr = 44100
            t = np.linspace(0, ALERT_DURATION, int(sr * ALERT_DURATION))
            audio = np.sin(2 * np.pi * 880 * t) * np.exp(-3 * t) * 0.3
            audio = (audio * 32767).astype(np.int16)
            wavfile.write(str(ALERT_SOUND_PATH), sr, audio)
            logger.info("✅ Archivo de alerta creado")
        except Exception as e:
            logger.warning(f"No se pudo crear audio: {e}")
    
    def _init_twilio(self):
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
            logger.info("✅ Twilio inicializado")
            return client
        except Exception as e:
            logger.warning(f"Twilio no disponible: {e}")
            return None
    
    def play_alert(self):
        if self.audio:
            try:
                self.audio.play()
                logger.info("🔊 Alerta reproducida")
            except Exception as e:
                logger.error(f"Error audio: {e}")
    
    def send_whatsapp(self, confidence):
        if not self.twilio:
            return False
        
        if self.last_alert:
            elapsed = (datetime.now() - self.last_alert).total_seconds()
            if elapsed < ALERT_COOLDOWN:
                return False
        
        try:
            now = datetime.now()
            msg = WHATSAPP_MESSAGE_TEMPLATE.format(
                fecha=now.strftime("%d/%m/%Y"),
                hora=now.strftime("%H:%M:%S"),
                ubicacion=DEFAULT_LOCATION,
                confianza=int(confidence * 100),
                num_personas=1
            )
            
            message = self.twilio.messages.create(
                body=msg,
                from_=f'whatsapp:{TWILIO_PHONE_NUMBER}',
                to=f'whatsapp:{DESTINATION_PHONE}'
            )
            
            self.last_alert = datetime.now()
            self.alerts_sent += 1
            logger.info(f"✅ WhatsApp enviado: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"Error WhatsApp: {e}")
            return False
    
    def save_frame(self, frame, is_alert=False):
        if not SAVE_DETECTIONS:
            return
        if SAVE_ONLY_ALERTS and not is_alert:
            return
        
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            prefix = "ALERTA" if is_alert else "det"
            path = DETECTIONS_DIR / f"{prefix}_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            if is_alert:
                logger.info(f"📸 Captura guardada: {path.name}")
        except Exception as e:
            logger.error(f"Error guardando: {e}")
    
    def calc_fps(self):
        self.fps_frames += 1
        elapsed = time.time() - self.fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_frames / elapsed
            self.fps_frames = 0
            self.fps_time = time.time()
        return self.fps
    
    def detect_faces(self, frame):
        """Detecta rostros en el frame"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return faces
    
    def process_frame(self, frame):
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # Detectar cascos
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, 
                           iou=IOU_THRESHOLD, verbose=False)
        
        helmet_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BOX_AREA:
                    continue
                
                helmet_boxes.append((x1, y1, x2, y2, conf, class_name))
                self.total_detections += 1
                
                # Dibujar casco detectado (VERDE)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WITH_HELMET, 3)
                label = f"{class_name} {conf:.0%}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WITH_HELMET, 2)
        
        # Detectar rostros
        faces = self.detect_faces(frame)
        faces_without_helmet = []
        
        for (fx, fy, fw, fh) in faces:
            face_center_y = fy + fh // 2
            face_center_x = fx + fw // 2
            
            # Verificar si este rostro tiene un casco cerca (arriba)
            has_helmet = False
            for (hx1, hy1, hx2, hy2, conf, name) in helmet_boxes:
                # El casco debe estar arriba del rostro
                helmet_center_x = (hx1 + hx2) // 2
                
                # Verificar alineación horizontal y que el casco esté arriba
                if abs(helmet_center_x - face_center_x) < fw and hy2 < face_center_y + 50:
                    has_helmet = True
                    break
            
            if not has_helmet:
                faces_without_helmet.append((fx, fy, fw, fh))
                # Dibujar rostro SIN casco (ROJO)
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), COLOR_WITHOUT_HELMET, 3)
                cv2.putText(frame, "SIN CASCO!", (fx, fy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WITHOUT_HELMET, 2)
        
        # Lógica de alertas
        person_without_helmet = len(faces_without_helmet) > 0
        
        if person_without_helmet:
            self.frames_no_helmet += 1
            self._draw_warning(frame)
            
            if self.frames_no_helmet == FRAMES_THRESHOLD:
                logger.warning("🚨 ALERTA: Persona SIN CASCO detectada")
                self.play_alert()
                self.send_whatsapp(0.9)
                self.save_frame(frame, is_alert=True)
        else:
            if self.frames_no_helmet > 0:
                logger.info("✅ Situación normalizada - Casco detectado")
            self.frames_no_helmet = 0
        
        # Panel de información
        self._draw_info(frame, len(helmet_boxes), len(faces_without_helmet))
        
        return frame
    
    def _draw_warning(self, frame):
        h, w = frame.shape[:2]
        text = "¡ALERTA! PERSONA SIN CASCO"
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        tx = (w - tw) // 2
        ty = 50
        
        # Parpadeo
        alpha = 0.8 if (self.frame_count // 8) % 2 == 0 else 0.5
        overlay = frame.copy()
        cv2.rectangle(overlay, (tx - 20, ty - 40), (tx + tw + 20, ty + 20), 
                     COLOR_WITHOUT_HELMET, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                   COLOR_TEXT, 3)
        
        # Contador
        counter = f"Frames: {self.frames_no_helmet}/{FRAMES_THRESHOLD}"
        cv2.putText(frame, counter, (tx + 50, ty + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    
    def _draw_info(self, frame, cascos, sin_casco):
        h, w = frame.shape[:2]
        
        # Panel inferior
        panel_h = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        fps = self.calc_fps()
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        
        # Estado
        if sin_casco > 0:
            estado = f"⚠️ {sin_casco} SIN CASCO"
            estado_color = COLOR_WITHOUT_HELMET
        elif cascos > 0:
            estado = f"✅ {cascos} CON CASCO"
            estado_color = COLOR_WITH_HELMET
        else:
            estado = "👀 Buscando..."
            estado_color = COLOR_WARNING
        
        y = h - panel_h + 30
        cv2.putText(frame, estado, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, estado_color, 2)
        cv2.putText(frame, f"Alertas: {self.alerts_sent}", (10, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        
        # Derecha
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 150, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        cv2.putText(frame, f"Tiempo: {mins:02d}:{secs:02d}", (w - 150, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        
        # Cooldown
        if self.last_alert:
            remaining = ALERT_COOLDOWN - (datetime.now() - self.last_alert).total_seconds()
            if remaining > 0:
                cv2.putText(frame, f"Cooldown: {remaining:.0f}s", 
                           (w//2 - 60, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, COLOR_WARNING, 2)
    
    def run(self):
        logger.info("🎥 Iniciando captura...")
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir cámara {CAMERA_INDEX}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"✅ Cámara: {w}x{h}")
        logger.info("")
        logger.info("🎮 CONTROLES:")
        logger.info("   Q/ESC - Salir")
        logger.info("   S     - Guardar screenshot")
        logger.info("   R     - Resetear alertas")
        logger.info("   P     - Pausar")
        logger.info("")
        
        paused = False
        window_name = 'Deteccion de Cascos - SENATI (Q para salir)'
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Error capturando frame")
                        break
                    
                    processed = self.process_frame(frame)
                else:
                    processed = frame.copy()
                    cv2.putText(processed, "PAUSADO - Presiona P", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, COLOR_WARNING, 2)
                
                cv2.imshow(window_name, processed)
                
                # Esperar tecla (importante: waitKey debe ser > 0)
                key = cv2.waitKey(1) & 0xFF
                
                # Múltiples formas de salir
                if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
                    logger.info("Saliendo...")
                    break
                elif key == ord('r') or key == ord('R'):
                    self.frames_no_helmet = 0
                    self.last_alert = None
                    logger.info("✅ Alertas reseteadas")
                elif key == ord('s') or key == ord('S'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{ts}.jpg"
                    cv2.imwrite(filename, processed)
                    logger.info(f"📸 Screenshot: {filename}")
                elif key == ord('p') or key == ord('P'):
                    paused = not paused
                    logger.info(f"{'⏸️ Pausado' if paused else '▶️ Reanudado'}")
                
                # Verificar si la ventana fue cerrada con X
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrumpido por usuario")
        
        finally:
            total_time = time.time() - self.start_time
            logger.info("")
            logger.info("=" * 60)
            logger.info("📊 ESTADÍSTICAS FINALES")
            logger.info(f"   Tiempo: {total_time:.1f}s")
            logger.info(f"   Frames: {self.frame_count}")
            logger.info(f"   FPS promedio: {self.frame_count / max(total_time, 1):.1f}")
            logger.info(f"   Detecciones cascos: {self.total_detections}")
            logger.info(f"   Alertas enviadas: {self.alerts_sent}")
            logger.info("=" * 60)
            
            cap.release()
            cv2.destroyAllWindows()
            # Forzar cierre de ventanas en Windows
            for _ in range(5):
                cv2.waitKey(1)


def main():
    print_config()
    
    print("\n" + "=" * 70)
    print("   Presiona Enter para iniciar o Ctrl+C para cancelar")
    print("=" * 70)
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Cancelado")
        return
    
    try:
        system = HelmetDetectionSystem()
        system.run()
    except Exception as e:
        logger.critical(f"Error crítico: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()