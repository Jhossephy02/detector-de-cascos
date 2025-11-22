# config.py
"""
Configuración del Sistema de Detección de Cascos
Versión Mejorada con validación completa
"""

import os
import sys
from pathlib import Path

# Intentar cargar dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv es opcional

# ==================== RUTAS DEL PROYECTO ====================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
ASSETS_DIR = BASE_DIR / 'assets'
LOGS_DIR = BASE_DIR / 'logs'
DATA_DIR = BASE_DIR / 'data'

# Crear directorios
for d in [MODELS_DIR, ASSETS_DIR, LOGS_DIR, DATA_DIR, DATA_DIR / 'detections']:
    d.mkdir(exist_ok=True)

# ==================== MODELO ====================
MODEL_PATH = MODELS_DIR / 'best.pt'
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
IMG_SIZE = int(os.getenv('IMG_SIZE', '640'))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.45'))

# Clases (se actualizan automáticamente al cargar el modelo)
CLASS_NAMES = {
    0: 'CASCO',
    1: 'SIN_CASCO'
}

# ==================== TWILIO/WHATSAPP ====================
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE', '+14155238886')
DESTINATION_PHONE = os.getenv('DESTINATION_PHONE', '')
ENABLE_WHATSAPP = os.getenv('ENABLE_WHATSAPP', 'false').lower() == 'true'

# ==================== AUDIO ====================
ALERT_SOUND_PATH = ASSETS_DIR / 'alert.wav'
ALERT_VOLUME = float(os.getenv('ALERT_VOLUME', '0.7'))
ALERT_DURATION = float(os.getenv('ALERT_DURATION', '1.5'))
ENABLE_AUDIO = os.getenv('ENABLE_AUDIO', 'true').lower() == 'true'

# ==================== DETECCIÓN ====================
FRAMES_THRESHOLD = int(os.getenv('FRAMES_THRESHOLD', '10'))
ALERT_COOLDOWN = int(os.getenv('ALERT_COOLDOWN', '30'))
TARGET_FPS = int(os.getenv('TARGET_FPS', '30'))
MIN_BOX_AREA = int(os.getenv('MIN_BOX_AREA', '2000'))

# ==================== CÁMARA ====================
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '1280'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '720'))
CAMERA_BACKEND = os.getenv('CAMERA_BACKEND', None)

# ==================== VISUALIZACIÓN ====================
COLOR_WITH_HELMET = (0, 255, 0)      # Verde
COLOR_WITHOUT_HELMET = (0, 0, 255)   # Rojo
COLOR_WARNING = (0, 165, 255)        # Naranja
COLOR_TEXT = (255, 255, 255)         # Blanco
COLOR_BACKGROUND = (0, 0, 0)         # Negro
BOX_THICKNESS = 2
TEXT_THICKNESS = 2
FONT_SCALE = 0.6
FONT = 0
SHOW_FPS = True
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# ==================== MENSAJES ====================
WHATSAPP_MESSAGE_TEMPLATE = """
🚨 ALERTA DE SEGURIDAD

⚠️ Persona SIN CASCO detectada

📅 {fecha}
🕐 {hora}
📍 {ubicacion}
🎯 Confianza: {confianza}%

⚡ Verifique el área inmediatamente.

---
Sistema de Vigilancia SENATI
""".strip()

DEFAULT_LOCATION = os.getenv('LOCATION', 'Área de construcción')

# ==================== LOGS ====================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = LOGS_DIR / 'detecciones.log'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_MAX_SIZE_MB = int(os.getenv('LOG_MAX_SIZE_MB', '10'))
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

# ==================== GUARDADO ====================
SAVE_DETECTIONS = os.getenv('SAVE_DETECTIONS', 'false').lower() == 'true'
DETECTIONS_DIR = DATA_DIR / 'detections'
SAVE_ONLY_ALERTS = os.getenv('SAVE_ONLY_ALERTS', 'true').lower() == 'true'

# ==================== AVANZADO ====================
INFERENCE_MODE = os.getenv('INFERENCE_MODE', 'realtime')
USE_HALF_PRECISION = os.getenv('USE_HALF_PRECISION', 'false').lower() == 'true'
ENABLE_TRACKING = os.getenv('ENABLE_TRACKING', 'false').lower() == 'true'
TRACKER_TYPE = os.getenv('TRACKER_TYPE', 'bytetrack.yaml')

# ==================== ENTRENAMIENTO (Colab) ====================
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'workers': 4,
    'patience': 50,
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
}


def validate_config():
    """Valida la configuración y retorna errores/advertencias"""
    errors, warnings = [], []
    
    # Modelo
    if not MODEL_PATH.exists():
        errors.append(f"Modelo no encontrado: {MODEL_PATH}")
    
    # Twilio
    if ENABLE_WHATSAPP:
        if not TWILIO_ACCOUNT_SID or 'tu_' in TWILIO_ACCOUNT_SID:
            warnings.append("TWILIO_ACCOUNT_SID no configurado")
        if not TWILIO_AUTH_TOKEN or 'tu_' in TWILIO_AUTH_TOKEN:
            warnings.append("TWILIO_AUTH_TOKEN no configurado")
        if not DESTINATION_PHONE:
            warnings.append("DESTINATION_PHONE no configurado")
    
    # Rangos
    if not 0 <= CONFIDENCE_THRESHOLD <= 1:
        errors.append(f"CONFIDENCE_THRESHOLD inválido: {CONFIDENCE_THRESHOLD}")
    if not 0 <= ALERT_VOLUME <= 1:
        errors.append(f"ALERT_VOLUME inválido: {ALERT_VOLUME}")
    
    return errors, warnings


def print_config():
    """Imprime la configuración actual"""
    print("=" * 70)
    print("🛡️  CONFIGURACIÓN DEL SISTEMA".center(70))
    print("=" * 70)
    
    sections = [
        ("📁 RUTAS", [
            f"Base: {BASE_DIR}",
            f"Modelos: {MODELS_DIR}",
            f"Logs: {LOGS_DIR}",
        ]),
        ("🤖 MODELO", [
            f"Ruta: {MODEL_PATH}",
            f"Existe: {'✅' if MODEL_PATH.exists() else '❌'}",
            f"Confianza: {CONFIDENCE_THRESHOLD}",
            f"IOU: {IOU_THRESHOLD}",
        ]),
        ("📱 WHATSAPP", [
            f"Habilitado: {'✅' if ENABLE_WHATSAPP else '❌'}",
            f"Destino: {DESTINATION_PHONE or 'No configurado'}",
        ]),
        ("🔊 AUDIO", [
            f"Habilitado: {'✅' if ENABLE_AUDIO else '❌'}",
            f"Volumen: {ALERT_VOLUME}",
        ]),
        ("🎥 CÁMARA", [
            f"Índice: {CAMERA_INDEX}",
            f"Resolución: {CAMERA_WIDTH}x{CAMERA_HEIGHT}",
        ]),
        ("⚙️ DETECCIÓN", [
            f"Umbral frames: {FRAMES_THRESHOLD}",
            f"Cooldown: {ALERT_COOLDOWN}s",
            f"Área mínima: {MIN_BOX_AREA}px²",
        ]),
    ]
    
    for title, items in sections:
        print(f"\n{title}")
        for item in items:
            print(f"   {item}")
    
    print("\n" + "=" * 70)


def get_summary():
    """Retorna resumen de configuración"""
    return {
        'model_exists': MODEL_PATH.exists(),
        'whatsapp': ENABLE_WHATSAPP,
        'audio': ENABLE_AUDIO,
        'tracking': ENABLE_TRACKING,
        'debug': DEBUG_MODE,
    }


if __name__ == "__main__":
    print_config()
    
    errors, warnings = validate_config()
    
    if warnings:
        print("\n⚠️  ADVERTENCIAS:")
        for w in warnings:
            print(f"   • {w}")
    
    if errors:
        print("\n❌ ERRORES:")
        for e in errors:
            print(f"   • {e}")
    else:
        print("\n✅ Configuración válida")