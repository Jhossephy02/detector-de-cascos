# config.py
"""
Configuración del Sistema de Detección de Cascos
Versión Mejorada con validación y configuración avanzada
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# ==================== RUTAS DEL PROYECTO ====================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
ASSETS_DIR = BASE_DIR / 'assets'
LOGS_DIR = BASE_DIR / 'logs'
DATA_DIR = BASE_DIR / 'data'

# Crear directorios si no existen
for directory in [MODELS_DIR, ASSETS_DIR, LOGS_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== CONFIGURACIÓN DEL MODELO ====================
# Ruta al modelo entrenado YOLOv8
MODEL_PATH = MODELS_DIR / 'best.pt'

# Confianza mínima para detecciones (0.0 - 1.0)
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

# Tamaño de imagen para inferencia
IMG_SIZE = int(os.getenv('IMG_SIZE', '640'))

# IOU threshold para NMS (Non-Maximum Suppression)
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.45'))

# Clases del modelo (adaptable según tu modelo)
CLASS_NAMES = {
    0: 'CASCO',  # o 'con_casco'
    # 1: 'sin_casco'  # Descomenta si tienes 2 clases
}

# ==================== CONFIGURACIÓN DE TWILIO ====================
# IMPORTANTE: Configura estas variables con tus credenciales de Twilio
# Obtén tus credenciales en: https://console.twilio.com/

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'tu_account_sid_aqui')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'tu_auth_token_aqui')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE', '+14155238886')  # Número de Twilio

# Número de destino (formato internacional: +51 para Perú)
DESTINATION_PHONE = os.getenv('DESTINATION_PHONE', '+51987654321')

# Habilitar/deshabilitar envío de mensajes
ENABLE_WHATSAPP = os.getenv('ENABLE_WHATSAPP', 'False').lower() == 'true'

# ==================== CONFIGURACIÓN DE AUDIO ====================
# Ruta al archivo de alerta
ALERT_SOUND_PATH = ASSETS_DIR / 'alert.wav'

# Volumen de la alerta (0.0 - 1.0)
ALERT_VOLUME = float(os.getenv('ALERT_VOLUME', '0.7'))

# Duración del sonido en segundos
ALERT_DURATION = float(os.getenv('ALERT_DURATION', '1.5'))

# Habilitar/deshabilitar audio
ENABLE_AUDIO = os.getenv('ENABLE_AUDIO', 'True').lower() == 'true'

# ==================== CONFIGURACIÓN DE DETECCIÓN ====================
# Número de frames consecutivos para confirmar detección
FRAMES_THRESHOLD = int(os.getenv('FRAMES_THRESHOLD', '10'))

# Cooldown entre alertas (segundos)
ALERT_COOLDOWN = int(os.getenv('ALERT_COOLDOWN', '30'))

# FPS objetivo de la cámara
TARGET_FPS = int(os.getenv('TARGET_FPS', '30'))

# Área mínima del bounding box (en píxeles²)
MIN_BOX_AREA = int(os.getenv('MIN_BOX_AREA', '2000'))

# ==================== CONFIGURACIÓN DE CÁMARA ====================
# Índice de la cámara (0 = cámara principal, 1 = cámara secundaria, etc.)
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))

# Resolución de la cámara
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '1280'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '720'))

# Backend de captura (opcional: cv2.CAP_DSHOW en Windows puede ser más rápido)
CAMERA_BACKEND = os.getenv('CAMERA_BACKEND', None)

# ==================== CONFIGURACIÓN DE VISUALIZACIÓN ====================
# Colores en formato BGR
COLOR_WITH_HELMET = (0, 255, 0)      # Verde
COLOR_WITHOUT_HELMET = (0, 0, 255)   # Rojo
COLOR_WARNING = (0, 165, 255)        # Naranja
COLOR_TEXT = (255, 255, 255)         # Blanco
COLOR_BACKGROUND = (0, 0, 0)         # Negro

# Grosor de líneas
BOX_THICKNESS = 2
TEXT_THICKNESS = 2

# Tamaño de fuente
FONT_SCALE = 0.6
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX

# Mostrar FPS en pantalla
SHOW_FPS = True

# Mostrar información de debug
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# ==================== CONFIGURACIÓN DE MENSAJES ====================
# Plantilla del mensaje de WhatsApp
WHATSAPP_MESSAGE_TEMPLATE = """
🚨 ALERTA DE SEGURIDAD - DETECCIÓN DE CASCO

⚠️ Se ha detectado una persona SIN CASCO DE SEGURIDAD en el área de trabajo.

📅 Fecha: {fecha}
🕐 Hora: {hora}
📍 Ubicación: {ubicacion}
🎯 Nivel de confianza: {confianza}%
👥 Personas detectadas: {num_personas}

⚡ ACCIÓN REQUERIDA:
Por favor, verifique el área inmediatamente y asegúrese de que todo el personal use el equipo de protección requerido.

---
Sistema automático de vigilancia de seguridad
SENATI - Ingeniería Civil
""".strip()

# Ubicación por defecto (personalízalo)
DEFAULT_LOCATION = os.getenv('LOCATION', 'Área de construcción')

# ==================== CONFIGURACIÓN DE LOGS ====================
# Nivel de logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Archivo de logs
LOG_FILE = LOGS_DIR / 'detecciones.log'

# Formato de logs
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Rotación de logs (tamaño máximo en MB)
LOG_MAX_SIZE_MB = int(os.getenv('LOG_MAX_SIZE_MB', '10'))
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

# ==================== CONFIGURACIÓN DE GUARDADO ====================
# Guardar frames con detecciones
SAVE_DETECTIONS = os.getenv('SAVE_DETECTIONS', 'False').lower() == 'true'
DETECTIONS_DIR = DATA_DIR / 'detections'
DETECTIONS_DIR.mkdir(exist_ok=True)

# Guardar solo frames con alertas (sin casco)
SAVE_ONLY_ALERTS = os.getenv('SAVE_ONLY_ALERTS', 'True').lower() == 'true'

# ==================== CONFIGURACIÓN DE ENTRENAMIENTO (Para Colab) ====================
# Configuración para entrenamiento en Google Colab
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'workers': 4,
    'patience': 50,  # Early stopping
    'optimizer': 'SGD',
    'lr0': 0.01,  # Learning rate inicial
    'lrf': 0.01,  # Learning rate final
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,  # Box loss gain
    'cls': 0.5,  # Class loss gain
    'dfl': 1.5,  # DFL loss gain
    'plots': True,
    'save_period': 10,  # Guardar cada X épocas
}

# ==================== CONFIGURACIÓN AVANZADA ====================
# Modo de inferencia
INFERENCE_MODE = os.getenv('INFERENCE_MODE', 'realtime')  # 'realtime' o 'batch'

# Usar half precision (FP16) para inferencia más rápida (requiere GPU)
USE_HALF_PRECISION = os.getenv('USE_HALF_PRECISION', 'False').lower() == 'true'

# Tracker (seguimiento de objetos)
ENABLE_TRACKING = os.getenv('ENABLE_TRACKING', 'False').lower() == 'true'
TRACKER_TYPE = os.getenv('TRACKER_TYPE', 'bytetrack.yaml')

# ==================== VALIDACIÓN DE CONFIGURACIÓN ====================
def validate_config():
    """Valida que la configuración sea correcta"""
    errors = []
    warnings = []
    
    # Validar modelo
    if not MODEL_PATH.exists():
        errors.append(f"⚠️ Modelo no encontrado en: {MODEL_PATH}")
        errors.append("   Descarga o entrena un modelo y colócalo en la carpeta 'models/'")
    
    # Validar Twilio
    if ENABLE_WHATSAPP:
        if 'tu_account_sid_aqui' in TWILIO_ACCOUNT_SID:
            warnings.append("⚠️ Configura TWILIO_ACCOUNT_SID en config.py o .env")
        if 'tu_auth_token_aqui' in TWILIO_AUTH_TOKEN:
            warnings.append("⚠️ Configura TWILIO_AUTH_TOKEN en config.py o .env")
        if '+51987654321' in DESTINATION_PHONE:
            warnings.append("⚠️ Actualiza DESTINATION_PHONE con tu número real")
    
    # Validar rangos
    if not 0 <= CONFIDENCE_THRESHOLD <= 1:
        errors.append(f"⚠️ CONFIDENCE_THRESHOLD debe estar entre 0 y 1 (actual: {CONFIDENCE_THRESHOLD})")
    
    if not 0 <= ALERT_VOLUME <= 1:
        errors.append(f"⚠️ ALERT_VOLUME debe estar entre 0 y 1 (actual: {ALERT_VOLUME})")
    
    # Validar audio
    if ENABLE_AUDIO and not ALERT_SOUND_PATH.exists():
        warnings.append(f"⚠️ Archivo de audio no encontrado: {ALERT_SOUND_PATH}")
        warnings.append("   Se creará automáticamente al ejecutar")
    
    return errors, warnings

# ==================== INFORMACIÓN DEL SISTEMA ====================
def print_config():
    """Imprime la configuración actual"""
    print("=" * 80)
    print("🛡️  CONFIGURACIÓN DEL SISTEMA DE DETECCIÓN DE CASCOS".center(80))
    print("=" * 80)
    
    print(f"\n{'📁 DIRECTORIOS':<30}")
    print(f"{'   Base:':<25} {BASE_DIR}")
    print(f"{'   Modelos:':<25} {MODELS_DIR}")
    print(f"{'   Assets:':<25} {ASSETS_DIR}")
    print(f"{'   Logs:':<25} {LOGS_DIR}")
    print(f"{'   Data:':<25} {DATA_DIR}")
    
    print(f"\n{'🤖 MODELO':<30}")
    print(f"{'   Ruta:':<25} {MODEL_PATH}")
    print(f"{'   Existe:':<25} {'✅ Sí' if MODEL_PATH.exists() else '❌ No'}")
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024*1024)
        print(f"{'   Tamaño:':<25} {size_mb:.1f} MB")
    print(f"{'   Confianza mínima:':<25} {CONFIDENCE_THRESHOLD}")
    print(f"{'   Tamaño de imagen:':<25} {IMG_SIZE}")
    print(f"{'   IOU threshold:':<25} {IOU_THRESHOLD}")
    
    print(f"\n{'📱 WHATSAPP/TWILIO':<30}")
    print(f"{'   Habilitado:':<25} {'✅ Sí' if ENABLE_WHATSAPP else '❌ No'}")
    if ENABLE_WHATSAPP:
        print(f"{'   Account SID:':<25} {TWILIO_ACCOUNT_SID[:20]}...")
        print(f"{'   Teléfono destino:':<25} {DESTINATION_PHONE}")
    
    print(f"\n{'🔊 AUDIO':<30}")
    print(f"{'   Habilitado:':<25} {'✅ Sí' if ENABLE_AUDIO else '❌ No'}")
    print(f"{'   Volumen:':<25} {ALERT_VOLUME}")
    print(f"{'   Duración:':<25} {ALERT_DURATION}s")
    
    print(f"\n{'🎥 CÁMARA':<30}")
    print(f"{'   Índice:':<25} {CAMERA_INDEX}")
    print(f"{'   Resolución:':<25} {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"{'   FPS objetivo:':<25} {TARGET_FPS}")
    
    print(f"\n{'⚙️  DETECCIÓN':<30}")
    print(f"{'   Umbral de frames:':<25} {FRAMES_THRESHOLD}")
    print(f"{'   Cooldown alertas:':<25} {ALERT_COOLDOWN}s")
    print(f"{'   Área mínima box:':<25} {MIN_BOX_AREA}px²")
    print(f"{'   Tracking:':<25} {'✅ Sí' if ENABLE_TRACKING else '❌ No'}")
    
    print(f"\n{'💾 GUARDADO':<30}")
    print(f"{'   Guardar detecciones:':<25} {'✅ Sí' if SAVE_DETECTIONS else '❌ No'}")
    if SAVE_DETECTIONS:
        print(f"{'   Solo alertas:':<25} {'✅ Sí' if SAVE_ONLY_ALERTS else '❌ No'}")
        print(f"{'   Directorio:':<25} {DETECTIONS_DIR}")
    
    print(f"\n{'📊 LOGS':<30}")
    print(f"{'   Nivel:':<25} {LOG_LEVEL}")
    print(f"{'   Archivo:':<25} {LOG_FILE}")
    print(f"{'   Tamaño máximo:':<25} {LOG_MAX_SIZE_MB} MB")
    
    print(f"\n{'🔧 AVANZADO':<30}")
    print(f"{'   Modo debug:':<25} {'✅ Sí' if DEBUG_MODE else '❌ No'}")
    print(f"{'   Half precision:':<25} {'✅ Sí' if USE_HALF_PRECISION else '❌ No'}")
    print(f"{'   Mostrar FPS:':<25} {'✅ Sí' if SHOW_FPS else '❌ No'}")
    
    print("\n" + "=" * 80)

def get_config_summary():
    """Retorna un diccionario con resumen de configuración"""
    return {
        'model_exists': MODEL_PATH.exists(),
        'whatsapp_enabled': ENABLE_WHATSAPP,
        'audio_enabled': ENABLE_AUDIO,
        'tracking_enabled': ENABLE_TRACKING,
        'save_detections': SAVE_DETECTIONS,
        'debug_mode': DEBUG_MODE,
    }

if __name__ == "__main__":
    print_config()
    errors, warnings = validate_config()
    
    if warnings:
        print("\n⚠️  ADVERTENCIAS:")
        for warning in warnings:
            print(f"   {warning}")
    
    if errors:
        print("\n❌ ERRORES DE CONFIGURACIÓN:")
        for error in errors:
            print(f"   {error}")
    else:
        print("\n✅ Configuración válida")
    
    print("\n💡 TIP: Crea un archivo .env para configurar variables de entorno")
    print("   Ejemplo: CONFIDENCE_THRESHOLD=0.6")