# main.py
"""
Sistema de Detección de Cascos de Seguridad con YOLOv8
Actividad 3 - SENATI
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pygame
from twilio.rest import Client
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Importar configuración
try:
    from config import *
except ImportError:
    print("❌ Error: No se encontró config.py")
    print("Asegúrate de tener el archivo config.py en la misma carpeta")
    sys.exit(1)

# ==================== CONFIGURACIÓN DE LOGGING ====================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CLASE PRINCIPAL ====================
class HelmetDetectionSystem:
    """Sistema de detección de cascos de seguridad"""
    
    def __init__(self):
        """Inicializa el sistema"""
        logger.info("Inicializando sistema de detección...")
        
        # Validar configuración
        errors = validate_config()
        if errors:
            logger.error("Errores de configuración encontrados:")
            for error in errors:
                logger.error(error)
            if not MODEL_PATH.exists():
                logger.critical("Modelo no encontrado. Sistema detenido.")
                sys.exit(1)
        
        # Inicializar componentes
        self.model = self._load_model()
        self.audio_system = self._init_audio()
        self.twilio_client = self._init_twilio()
        
        # Variables de control
        self.frames_without_helmet = 0
        self.last_alert_time = None
        self.total_detections = 0
        self.alerts_sent = 0
        
        logger.info("✅ Sistema inicializado correctamente")
    
    def _load_model(self):
        """Carga el modelo YOLOv8"""
        try:
            logger.info(f"Cargando modelo desde: {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
            logger.info("✅ Modelo cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            sys.exit(1)
    
    def _init_audio(self):
        """Inicializa el sistema de audio"""
        try:
            pygame.mixer.init()
            
            # Crear sonido de alerta si no existe
            if not ALERT_SOUND_PATH.exists():
                logger.info("Creando archivo de alerta...")
                self._create_alert_sound()
            
            alert_sound = pygame.mixer.Sound(str(ALERT_SOUND_PATH))
            alert_sound.set_volume(ALERT_VOLUME)
            
            logger.info("✅ Sistema de audio inicializado")
            return alert_sound
        except Exception as e:
            logger.error(f"⚠️ Error inicializando audio: {e}")
            return None
    
    def _create_alert_sound(self):
        """Crea un sonido de alerta básico"""
        try:
            from scipy.io import wavfile
            
            sample_rate = 44100
            frequency = 880  # A5
            t = np.linspace(0, ALERT_DURATION, int(sample_rate * ALERT_DURATION))
            
            # Generar tono con envolvente
            envelope = np.exp(-3 * t)
            audio_data = np.sin(2 * np.pi * frequency * t) * envelope * 0.3
            audio_data = (audio_data * 32767).astype(np.int16)
            
            wavfile.write(str(ALERT_SOUND_PATH), sample_rate, audio_data)
            logger.info(f"✅ Archivo de alerta creado: {ALERT_SOUND_PATH}")
        except Exception as e:
            logger.error(f"❌ Error creando sonido: {e}")
    
    def _init_twilio(self):
        """Inicializa el cliente de Twilio"""
        if not ENABLE_WHATSAPP:
            logger.info("WhatsApp deshabilitado en configuración")
            return None
        
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            # Verificar credenciales
            client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
            logger.info("✅ Cliente de Twilio inicializado")
            return client
        except Exception as e:
            logger.warning(f"⚠️ Twilio no disponible: {e}")
            return None
    
    def play_alert(self):
        """Reproduce alerta de audio"""
        if self.audio_system:
            try:
                self.audio_system.play()
                logger.info("🔊 Alerta de audio reproducida")
            except Exception as e:
                logger.error(f"Error reproduciendo audio: {e}")
    
    def send_whatsapp_alert(self, confidence):
        """Envía alerta por WhatsApp"""
        if not self.twilio_client:
            logger.warning("Cliente de Twilio no disponible")
            return False
        
        # Verificar cooldown
        if self.last_alert_time:
            elapsed = (datetime.now() - self.last_alert_time).total_seconds()
            if elapsed < ALERT_COOLDOWN:
                logger.info(f"Cooldown activo. {ALERT_COOLDOWN - elapsed:.0f}s restantes")
                return False
        
        try:
            # Preparar mensaje
            now = datetime.now()
            mensaje = WHATSAPP_MESSAGE_TEMPLATE.format(
                fecha=now.strftime("%d/%m/%Y"),
                hora=now.strftime("%H:%M:%S"),
                confianza=int(confidence * 100)
            )
            
            # Enviar mensaje
            message = self.twilio_client.messages.create(
                body=mensaje,
                from_=f'whatsapp:{TWILIO_PHONE_NUMBER}',
                to=f'whatsapp:{DESTINATION_PHONE}'
            )
            
            self.last_alert_time = datetime.now()
            self.alerts_sent += 1
            
            logger.info(f"✅ Mensaje WhatsApp enviado: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje: {e}")
            return False
    
    def process_frame(self, frame):
        """Procesa un frame y realiza detección"""
        # Realizar inferencia
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        person_without_helmet = False
        max_confidence = 0
        
        # Procesar detecciones
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Obtener información de la detección
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Obtener nombre de clase
                class_name = self.model.names[cls]
                
                # Determinar si es sin casco
                is_without_helmet = (
                    'sin_casco' in class_name.lower() or 
                    'sin casco' in class_name.lower() or
                    'no_helmet' in class_name.lower() or
                    'without' in class_name.lower() or
                    cls == 1  # Asumiendo que clase 1 es sin casco
                )
                
                if is_without_helmet:
                    person_without_helmet = True
                    max_confidence = max(max_confidence, conf)
                    color = COLOR_WITHOUT_HELMET
                else:
                    color = COLOR_WITH_HELMET
                
                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                
                # Etiqueta
                label = f'{class_name}: {conf:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                 FONT_SCALE, TEXT_THICKNESS)
                
                # Fondo de la etiqueta
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Texto de la etiqueta
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 
                           COLOR_TEXT, TEXT_THICKNESS)
                
                self.total_detections += 1
        
        # Gestionar alertas
        if person_without_helmet:
            self.frames_without_helmet += 1
            
            # Mostrar advertencia grande
            warning_text = '¡ALERTA! PERSONA SIN CASCO'
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       1.2, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            
            # Fondo de advertencia
            cv2.rectangle(frame, (text_x - 10, 10), 
                         (text_x + text_size[0] + 10, 50), 
                         (0, 0, 255), -1)
            cv2.putText(frame, warning_text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Activar alerta si supera umbral
            if self.frames_without_helmet >= FRAMES_THRESHOLD:
                if self.frames_without_helmet == FRAMES_THRESHOLD:  # Solo una vez
                    logger.warning("🚨 ALERTA: Persona sin casco detectada")
                    self.play_alert()
                    self.send_whatsapp_alert(max_confidence)
        else:
            self.frames_without_helmet = 0
        
        # Información en pantalla
        self._draw_info_panel(frame)
        
        return frame
    
    def _draw_info_panel(self, frame):
        """Dibuja panel de información en el frame"""
        h, w = frame.shape[:2]
        
        # Panel inferior
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Información
        info_texts = [
            f'Detecciones: {self.total_detections}',
            f'Alertas enviadas: {self.alerts_sent}',
            f'Frames sin casco: {self.frames_without_helmet}/{FRAMES_THRESHOLD}',
            f'FPS: {cv2.getTickFrequency() / (cv2.getTickCount()):.1f}'
        ]
        
        y_pos = h - panel_height + 25
        for text in info_texts:
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            y_pos += 20
    
    def run(self):
        """Ejecuta el sistema de detección"""
        logger.info("🎥 Iniciando captura de video...")
        
        # Abrir cámara
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            logger.error("❌ No se pudo abrir la cámara")
            return
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        logger.info("✅ Cámara iniciada correctamente")
        logger.info("Presiona 'q' para salir, 'r' para resetear alertas")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Error capturando frame")
                    break
                
                # Procesar frame
                processed_frame = self.process_frame(frame)
                
                # Mostrar
                cv2.imshow('Sistema de Detección de Cascos - SENATI', processed_frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Saliendo...")
                    break
                elif key == ord('r'):
                    self.frames_without_helmet = 0
                    self.last_alert_time = None
                    logger.info("Alertas reseteadas")
        
        except KeyboardInterrupt:
            logger.info("Interrupción por teclado")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("✅ Sistema finalizado correctamente")

# ==================== EJECUCIÓN PRINCIPAL ====================
def main():
    """Función principal"""
    print_config()
    
    try:
        system = HelmetDetectionSystem()
        system.run()
    except Exception as e:
        logger.critical(f"Error crítico: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()