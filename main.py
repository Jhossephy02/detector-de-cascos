# main.py
"""
Sistema de Detección de Cascos de Seguridad con YOLOv8
SENATI - Ingeniería Civil
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
    """Configura el sistema de logs"""
    logger = logging.getLogger('HelmetDetection')
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # Handler archivo
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    
    # Handler consola
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()


# ==================== SISTEMA PRINCIPAL ====================
class HelmetDetectionSystem:
    """Sistema de detección de cascos"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Iniciando Sistema de Detección de Cascos")
        logger.info("=" * 60)
        
        # Validar configuración
        errors, warnings = validate_config()
        for w in warnings:
            logger.warning(w)
        if errors:
            for e in errors:
                logger.error(e)
            if not MODEL_PATH.exists():
                logger.critical("Modelo no encontrado. Abortando.")
                sys.exit(1)
        
        # Componentes
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
        
        # FPS
        self.fps = 0
        self.fps_time = time.time()
        self.fps_frames = 0
        
        logger.info("✅ Sistema listo")
    
    def _load_model(self):
        """Carga el modelo YOLOv8"""
        try:
            from ultralytics import YOLO
            logger.info(f"Cargando modelo: {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
            logger.info(f"✅ Modelo cargado - Clases: {list(model.names.values())}")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            sys.exit(1)
    
    def _init_audio(self):
        """Inicializa el sistema de audio"""
        try:
            import pygame
            pygame.mixer.init()
            
            # Crear sonido si no existe
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
        """Crea un sonido de alerta básico"""
        try:
            from scipy.io import wavfile
            
            sr = 44100
            t = np.linspace(0, ALERT_DURATION, int(sr * ALERT_DURATION))
            freq = 880
            envelope = np.exp(-3 * t)
            audio = np.sin(2 * np.pi * freq * t) * envelope * 0.3
            audio = (audio * 32767).astype(np.int16)
            
            wavfile.write(str(ALERT_SOUND_PATH), sr, audio)
            logger.info(f"✅ Archivo de alerta creado")
        except Exception as e:
            logger.warning(f"No se pudo crear audio: {e}")
    
    def _init_twilio(self):
        """Inicializa Twilio para WhatsApp"""
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
        """Reproduce alerta de audio"""
        if self.audio:
            try:
                self.audio.play()
                logger.info("🔊 Alerta reproducida")
            except Exception as e:
                logger.error(f"Error audio: {e}")
    
    def send_whatsapp(self, confidence, count=1):
        """Envía alerta por WhatsApp"""
        if not self.twilio:
            return False
        
        # Cooldown
        if self.last_alert:
            elapsed = (datetime.now() - self.last_alert).total_seconds()
            if elapsed < ALERT_COOLDOWN:
                logger.debug(f"Cooldown: {ALERT_COOLDOWN - elapsed:.0f}s")
                return False
        
        try:
            now = datetime.now()
            msg = WHATSAPP_MESSAGE_TEMPLATE.format(
                fecha=now.strftime("%d/%m/%Y"),
                hora=now.strftime("%H:%M:%S"),
                ubicacion=DEFAULT_LOCATION,
                confianza=int(confidence * 100)
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
        """Guarda el frame"""
        if not SAVE_DETECTIONS:
            return
        if SAVE_ONLY_ALERTS and not is_alert:
            return
        
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            prefix = "alert" if is_alert else "det"
            path = DETECTIONS_DIR / f"{prefix}_{ts}.jpg"
            cv2.imwrite(str(path), frame)
        except Exception as e:
            logger.error(f"Error guardando: {e}")
    
    def calc_fps(self):
        """Calcula FPS"""
        self.fps_frames += 1
        elapsed = time.time() - self.fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_frames / elapsed
            self.fps_frames = 0
            self.fps_time = time.time()
        return self.fps
    
    def is_without_helmet(self, cls, name):
        """Determina si la detección es 'sin casco'"""
        name_low = name.lower()
        # Palabras clave que indican sin casco
        no_helmet_kw = ['sin', 'without', 'no_', 'no-', 'missing']
        return any(kw in name_low for kw in no_helmet_kw) or cls == 1
    
    def process_frame(self, frame):
        """Procesa un frame"""
        self.frame_count += 1
        
        # Inferencia
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, 
                           iou=IOU_THRESHOLD, verbose=False)
        
        person_no_helmet = False
        max_conf = 0
        det_count = 0
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Filtrar por área
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BOX_AREA:
                    continue
                
                class_name = self.model.names[cls]
                no_helmet = self.is_without_helmet(cls, class_name)
                
                det_count += 1
                
                if no_helmet:
                    person_no_helmet = True
                    max_conf = max(max_conf, conf)
                    color = COLOR_WITHOUT_HELMET
                else:
                    color = COLOR_WITH_HELMET
                
                # Dibujar
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                
                label = f"{class_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICKNESS)
                
                ly = y1 - 10 if y1 > 30 else y1 + th + 10
                cv2.rectangle(frame, (x1, ly - th - 5), 
                            (x1 + tw + 10, ly + 5), color, -1)
                cv2.putText(frame, label, (x1 + 5, ly), 
                           FONT, FONT_SCALE, COLOR_TEXT, TEXT_THICKNESS)
                
                self.total_detections += 1
        
        # Alertas
        if person_no_helmet:
            self.frames_no_helmet += 1
            self._draw_warning(frame)
            
            if self.frames_no_helmet == FRAMES_THRESHOLD:
                logger.warning("🚨 ALERTA: Persona sin casco")
                self.play_alert()
                self.send_whatsapp(max_conf, det_count)
                self.save_frame(frame, is_alert=True)
        else:
            if self.frames_no_helmet > 0:
                logger.info("✅ Situación normalizada")
            self.frames_no_helmet = 0
            if det_count > 0:
                self.save_frame(frame, is_alert=False)
        
        self._draw_info(frame, det_count)
        return frame
    
    def _draw_warning(self, frame):
        """Dibuja advertencia en pantalla"""
        h, w = frame.shape[:2]
        text = "¡ALERTA! PERSONA SIN CASCO"
        
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.2, 3)
        tx = (w - tw) // 2
        ty = 50
        
        # Parpadeo
        alpha = 0.7 if (self.frame_count // 10) % 2 == 0 else 0.5
        overlay = frame.copy()
        cv2.rectangle(overlay, (tx - 20, ty - 40), 
                     (tx + tw + 20, ty + 20), COLOR_WITHOUT_HELMET, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        cv2.putText(frame, text, (tx, ty), FONT, 1.2, COLOR_TEXT, 3)
        
        # Contador
        counter = f"{self.frames_no_helmet}/{FRAMES_THRESHOLD}"
        cv2.putText(frame, counter, (tx, ty + 35), FONT, 0.7, COLOR_TEXT, 2)
    
    def _draw_info(self, frame, det_count):
        """Panel de información"""
        h, w = frame.shape[:2]
        
        # Panel
        panel_h = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), COLOR_BACKGROUND, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        fps = self.calc_fps()
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        
        # Info izquierda
        y = h - panel_h + 25
        info_left = [
            f"Detecciones: {self.total_detections}",
            f"Actuales: {det_count}",
            f"Alertas: {self.alerts_sent}",
        ]
        for text in info_left:
            cv2.putText(frame, text, (10, y), FONT, 0.5, COLOR_TEXT, 1)
            y += 25
        
        # Info derecha
        y = h - panel_h + 25
        frame_color = COLOR_WITHOUT_HELMET if self.frames_no_helmet > 0 else COLOR_WITH_HELMET
        
        info_right = [
            (f"Sin casco: {self.frames_no_helmet}/{FRAMES_THRESHOLD}", frame_color),
            (f"FPS: {fps:.1f}", COLOR_TEXT),
            (f"Tiempo: {hrs:02d}:{mins:02d}:{secs:02d}", COLOR_TEXT),
        ]
        for text, color in info_right:
            cv2.putText(frame, text, (w // 2, y), FONT, 0.5, color, 1)
            y += 25
        
        # Cooldown
        if self.last_alert:
            remaining = ALERT_COOLDOWN - (datetime.now() - self.last_alert).total_seconds()
            if remaining > 0:
                cv2.putText(frame, f"Cooldown: {remaining:.0f}s", 
                           (10, h - 10), FONT, 0.5, COLOR_WARNING, 1)
    
    def run(self):
        """Ejecuta el sistema"""
        logger.info("🎥 Iniciando captura...")
        
        # Abrir cámara
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir cámara {CAMERA_INDEX}")
            logger.error("   Prueba CAMERA_INDEX=1 en .env")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"✅ Cámara: {w}x{h}")
        logger.info("")
        logger.info("🎮 CONTROLES:")
        logger.info("   Q - Salir | R - Reset | S - Screenshot | P - Pausa")
        logger.info("")
        
        paused = False
        
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
                               (10, 30), FONT, 0.8, COLOR_WARNING, 2)
                
                cv2.imshow('Detección de Cascos - SENATI', processed)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Saliendo...")
                    break
                elif key == ord('r'):
                    self.frames_no_helmet = 0
                    self.last_alert = None
                    logger.info("✅ Alertas reseteadas")
                elif key == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"screenshot_{ts}.jpg", processed)
                    logger.info(f"📸 Screenshot guardado")
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"{'⏸️ Pausado' if paused else '▶️ Reanudado'}")
        
        except KeyboardInterrupt:
            logger.info("Interrumpido")
        
        finally:
            # Estadísticas
            total_time = time.time() - self.start_time
            logger.info("")
            logger.info("=" * 60)
            logger.info("📊 ESTADÍSTICAS FINALES")
            logger.info(f"   Tiempo: {total_time:.1f}s")
            logger.info(f"   Frames: {self.frame_count}")
            logger.info(f"   FPS promedio: {self.frame_count / max(total_time, 1):.1f}")
            logger.info(f"   Detecciones: {self.total_detections}")
            logger.info(f"   Alertas: {self.alerts_sent}")
            logger.info("=" * 60)
            
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Función principal"""
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