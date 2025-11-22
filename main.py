# main.py
"""
Sistema de Detección de Cascos de Seguridad con YOLOv8
Versión Mejorada - SENATI
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pygame
from twilio.rest import Client
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import time

# Importar configuración
try:
    from config import *
except ImportError:
    print("❌ Error: No se encontró config.py")
    print("Asegúrate de tener el archivo config.py en la misma carpeta")
    sys.exit(1)

# ==================== CONFIGURACIÓN DE LOGGING ====================
def setup_logging():
    """Configura el sistema de logging con rotación"""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Handler para archivo con rotación
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ==================== CLASE PRINCIPAL ====================
class HelmetDetectionSystem:
    """Sistema de detección de cascos de seguridad"""
    
    def __init__(self):
        """Inicializa el sistema"""
        logger.info("="*60)
        logger.info("Inicializando Sistema de Detección de Cascos")
        logger.info("="*60)
        
        # Validar configuración
        errors, warnings = validate_config()
        
        if warnings:
            for warning in warnings:
                logger.warning(warning)
        
        if errors:
            logger.error("Errores de configuración encontrados:")
            for error in errors:
                logger.error(error)
            if not MODEL_PATH.exists():
                logger.critical("Modelo no encontrado. Sistema detenido.")
                sys.exit(1)
        
        # Inicializar componentes
        self.model = self._load_model()
        self.audio_system = self._init_audio() if ENABLE_AUDIO else None
        self.twilio_client = self._init_twilio() if ENABLE_WHATSAPP else None
        
        # Variables de control
        self.frames_without_helmet = 0
        self.last_alert_time = None
        self.total_detections = 0
        self.alerts_sent = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Para cálculo de FPS
        self.fps = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
        # Tracking de IDs (si está habilitado)
        self.tracked_ids = {}
        
        logger.info("✅ Sistema inicializado correctamente")
        logger.info("="*60)
    
    def _load_model(self):
        """Carga el modelo YOLOv8"""
        try:
            logger.info(f"Cargando modelo desde: {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
            
            # Configurar modelo
            if USE_HALF_PRECISION:
                model.to('cuda')
                model.half()
                logger.info("✅ Modelo configurado con half precision (FP16)")
            
            logger.info("✅ Modelo cargado exitosamente")
            logger.info(f"   Clases: {model.names}")
            
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
    
    def send_whatsapp_alert(self, confidence, num_personas=1):
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
                ubicacion=DEFAULT_LOCATION,
                confianza=int(confidence * 100),
                num_personas=num_personas
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
    
    def save_detection_frame(self, frame, alert=False):
        """Guarda el frame con detección"""
        if not SAVE_DETECTIONS:
            return
        
        if SAVE_ONLY_ALERTS and not alert:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{'alert' if alert else 'detection'}_{timestamp}.jpg"
            filepath = DETECTIONS_DIR / filename
            
            cv2.imwrite(str(filepath), frame)
            logger.debug(f"Frame guardado: {filename}")
        except Exception as e:
            logger.error(f"Error guardando frame: {e}")
    
    def calculate_fps(self):
        """Calcula los FPS actuales"""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:  # Actualizar cada segundo
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        
        return self.fps
    
    def detect_helmet_status(self, class_name, cls):
        """Determina si es sin casco basado en nombre o clase"""
        class_name_lower = class_name.lower()
        
        # Buscar palabras clave que indiquen "sin casco"
        without_keywords = ['sin', 'without', 'no', 'missing']
        
        for keyword in without_keywords:
            if keyword in class_name_lower:
                return True
        
        # Si el modelo solo tiene 1 clase (CASCO), invertir lógica
        if len(self.model.names) == 1:
            return False  # Si detecta CASCO, tiene casco
        
        # Si hay múltiples clases, asumir que clase 1 es sin casco
        if cls == 1:
            return True
        
        return False
    
    def process_frame(self, frame):
        """Procesa un frame y realiza detección"""
        self.frame_count += 1
        
        # Realizar inferencia
        if ENABLE_TRACKING:
            results = self.model.track(
                frame, 
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                persist=True,
                tracker=TRACKER_TYPE,
                verbose=False
            )
        else:
            results = self.model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )
        
        person_without_helmet = False
        max_confidence = 0
        detections_count = 0
        boxes_info = []
        
        # Procesar detecciones
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Obtener información de la detección
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Filtrar por área mínima
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < MIN_BOX_AREA:
                    continue
                
                # Obtener nombre de clase
                class_name = self.model.names[cls]
                
                # Determinar si es sin casco
                is_without_helmet = self.detect_helmet_status(class_name, cls)
                
                # ID de tracking (si está disponible)
                track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                
                detections_count += 1
                boxes_info.append({
                    'class': class_name,
                    'conf': conf,
                    'box': (x1, y1, x2, y2),
                    'without_helmet': is_without_helmet,
                    'track_id': track_id
                })
                
                if is_without_helmet:
                    person_without_helmet = True
                    max_confidence = max(max_confidence, conf)
                    color = COLOR_WITHOUT_HELMET
                else:
                    color = COLOR_WITH_HELMET
                
                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                
                # Preparar etiqueta
                label_parts = [class_name, f'{conf:.2f}']
                if track_id is not None:
                    label_parts.append(f'ID:{track_id}')
                label = ' | '.join(label_parts)
                
                # Calcular tamaño de etiqueta
                label_size, _ = cv2.getTextSize(
                    label, 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, 
                    TEXT_THICKNESS
                )
                
                # Fondo de la etiqueta
                label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
                cv2.rectangle(
                    frame, 
                    (x1, label_y - label_size[1] - 5),
                    (x1 + label_size[0] + 10, label_y + 5), 
                    color, 
                    -1
                )
                
                # Texto de la etiqueta
                cv2.putText(
                    frame, 
                    label, 
                    (x1 + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    FONT_SCALE,
                    COLOR_TEXT, 
                    TEXT_THICKNESS
                )
                
                self.total_detections += 1
        
        # Gestionar alertas
        if person_without_helmet:
            self.frames_without_helmet += 1
            
            # Mostrar advertencia grande
            self._draw_warning(frame)
            
            # Activar alerta si supera umbral
            if self.frames_without_helmet >= FRAMES_THRESHOLD:
                if self.frames_without_helmet == FRAMES_THRESHOLD:  # Solo una vez
                    logger.warning("🚨 ALERTA: Persona sin casco detectada")
                    self.play_alert()
                    self.send_whatsapp_alert(max_confidence, detections_count)
                    self.save_detection_frame(frame, alert=True)
        else:
            if self.frames_without_helmet > 0:
                logger.info(f"✅ Situación normalizada después de {self.frames_without_helmet} frames")
            self.frames_without_helmet = 0
            
            if detections_count > 0 and SAVE_DETECTIONS:
                self.save_detection_frame(frame, alert=False)
        
        # Información en pantalla
        self._draw_info_panel(frame, detections_count)
        
        return frame
    
    def _draw_warning(self, frame):
        """Dibuja advertencia en pantalla"""
        h, w = frame.shape[:2]
        
        warning_text = '¡ALERTA! PERSONA SIN CASCO'
        
        # Calcular tamaño del texto
        text_size = cv2.getTextSize(
            warning_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            3
        )[0]
        
        text_x = (w - text_size[0]) // 2
        text_y = 50
        
        # Fondo parpadeante
        alpha = 0.7 if (self.frame_count // 10) % 2 == 0 else 0.5
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (text_x - 20, text_y - 35),
            (text_x + text_size[0] + 20, text_y + 15),
            COLOR_WITHOUT_HELMET,
            -1
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Texto
        cv2.putText(
            frame,
            warning_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            COLOR_TEXT,
            3
        )
        
        # Contador de frames
        counter_text = f'{self.frames_without_helmet}/{FRAMES_THRESHOLD}'
        cv2.putText(
            frame,
            counter_text,
            (text_x, text_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            2
        )
    
    def _draw_info_panel(self, frame, detections_count):
        """Dibuja panel de información en el frame"""
        h, w = frame.shape[:2]
        
        # Panel inferior
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), COLOR_BACKGROUND, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Calcular FPS
        current_fps = self.calculate_fps()
        
        # Calcular tiempo de ejecución
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        # Información - Columna 1
        col1_x = 10
        y_pos = h - panel_height + 20
        
        info_col1 = [
            f'Detecciones totales: {self.total_detections}',
            f'Detecciones actuales: {detections_count}',
            f'Alertas enviadas: {self.alerts_sent}',
        ]
        
        for text in info_col1:
            cv2.putText(frame, text, (col1_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            y_pos += 25
        
        # Información - Columna 2
        col2_x = w // 2
        y_pos = h - panel_height + 20
        
        # Color del indicador de frames
        frame_color = COLOR_WITHOUT_HELMET if self.frames_without_helmet > 0 else COLOR_WITH_HELMET
        
        info_col2 = [
            (f'Frames sin casco: {self.frames_without_helmet}/{FRAMES_THRESHOLD}', frame_color),
            (f'FPS: {current_fps:.1f}', COLOR_TEXT),
            (f'Tiempo: {hours:02d}:{minutes:02d}:{seconds:02d}', COLOR_TEXT),
        ]
        
        for text, color in info_col2:
            cv2.putText(frame, text, (col2_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 25
        
        # Cooldown de alertas
        if self.last_alert_time:
            elapsed = (datetime.now() - self.last_alert_time).total_seconds()
            remaining = max(0, ALERT_COOLDOWN - elapsed)
            if remaining > 0:
                cooldown_text = f'Cooldown: {remaining:.0f}s'
                cv2.putText(frame, cooldown_text, (col1_x, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1)
    
    def run(self):
        """Ejecuta el sistema de detección"""
        logger.info("🎥 Iniciando captura de video...")
        
        # Abrir cámara
        if CAMERA_BACKEND:
            cap = cv2.VideoCapture(CAMERA_INDEX, getattr(cv2, CAMERA_BACKEND))
        else:
            cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            logger.error("❌ No se pudo abrir la cámara")
            logger.error(f"   Intenta cambiar CAMERA_INDEX en config.py (actual: {CAMERA_INDEX})")
            return
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        # Verificar configuración real
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info("✅ Cámara iniciada correctamente")
        logger.info(f"   Resolución: {actual_width}x{actual_height}")
        logger.info(f"   FPS: {actual_fps}")
        logger.info("")
        logger.info("🎮 CONTROLES:")
        logger.info("   'q' - Salir")
        logger.info("   'r' - Resetear alertas")
        logger.info("   's' - Guardar frame")
        logger.info("   'p' - Pausar/Reanudar")
        logger.info("   'd' - Toggle debug")
        logger.info("")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.error("Error capturando frame")
                        break
                    
                    # Procesar frame
                    processed_frame = self.process_frame(frame)
                else:
                    processed_frame = frame.copy()
                    cv2.putText(
                        processed_frame,
                        'PAUSADO - Presiona P para continuar',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        COLOR_WARNING,
                        2
                    )
                
                # Mostrar
                window_name = 'Sistema de Detección de Cascos - SENATI'
                cv2.imshow(window_name, processed_frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Saliendo...")
                    break
                elif key == ord('r'):
                    self.frames_without_helmet = 0
                    self.last_alert_time = None
                    logger.info("✅ Alertas reseteadas")
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    logger.info(f"📸 Screenshot guardado: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"{'⏸️  Pausado' if paused else '▶️  Reanudado'}")
                elif key == ord('d'):
                    global DEBUG_MODE
                    DEBUG_MODE = not DEBUG_MODE
                    logger.info(f"🔧 Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
        
        except KeyboardInterrupt:
            logger.info("Interrupción por teclado")
        
        finally:
            # Estadísticas finales
            total_time = time.time() - self.start_time
            logger.info("")
            logger.info("="*60)
            logger.info("📊 ESTADÍSTICAS FINALES")
            logger.info("="*60)
            logger.info(f"   Tiempo total: {total_time:.1f}s")
            logger.info(f"   Frames procesados: {self.frame_count}")
            logger.info(f"   FPS promedio: {self.frame_count / total_time:.1f}")
            logger.info(f"   Detecciones totales: {self.total_detections}")
            logger.info(f"   Alertas enviadas: {self.alerts_sent}")
            logger.info("="*60)
            
            cap.release()
            cv2.destroyAllWindows()
            logger.info("✅ Sistema finalizado correctamente")

# ==================== EJECUCIÓN PRINCIPAL ====================
def main():
    """Función principal"""
    # Mostrar configuración
    print_config()
    
    # Confirmar inicio
    print("\n" + "="*80)
    print("   ¿Iniciar sistema de detección? (Presiona Enter para continuar)")
    print("="*80)
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Inicio cancelado")
        sys.exit(0)
    
    try:
        system = HelmetDetectionSystem()
        system.run()
    except Exception as e:
        logger.critical(f"Error crítico: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()