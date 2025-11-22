# config.py
"""
Configuración del Sistema de Detección de Cascos
"""

import os
from pathlib import Path

# ==================== RUTAS DEL PROYECTO ====================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
ASSETS_DIR = BASE_DIR / 'assets'
LOGS_DIR = BASE_DIR / 'logs'

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ==================== CONFIGURACIÓN DEL MODELO ====================
# Ruta al modelo entrenado YOLOv8
MODEL_PATH = MODELS_DIR / 'best.pt'

# Confianza mínima para detecciones (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Tamaño de imagen para inferencia
IMG_SIZE = 640

# Clases del modelo
CLASS_NAMES = {
    0: 'con_casco',
    1: 'sin_casco'
}

# ==================== CONFIGURACIÓN DE TWILIO ====================
# IMPORTANTE: Configura estas variables con tus credenciales de Twilio
# Obtén tus credenciales en: https://console.twilio.com/

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'tu_account_sid_aqui')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'tu_auth_token_aqui')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE', '+14155238886')  # Número de Twilio Sandbox

# Número de destino (formato internacional: +51 para Perú)
DESTINATION_PHONE = os.getenv('DESTINATION_PHONE', '+51987654321')

# Habilitar/deshabilitar envío de mensajes
ENABLE_WHATSAPP = True  # Cambiar a False para desactivar

# ==================== CONFIGURACIÓN DE AUDIO ====================
# Ruta al archivo de alerta
ALERT_SOUND_PATH = ASSETS_DIR / 'alert.wav'

# Volumen de la alerta (0.0 - 1.0)
ALERT_VOLUME = 0.7

# Duración del sonido en segundos
ALERT_DURATION = 1.5

# ==================== CONFIGURACIÓN DE DETECCIÓN ====================
# Número de frames consecutivos para confirmar detección
FRAMES_THRESHOLD = 10

# Cooldown entre alertas (segundos)
ALERT_COOLDOWN = 30

# FPS objetivo de la cámara
TARGET_FPS = 30

# ==================== CONFIGURACIÓN DE CÁMARA ====================
# Índice de la cámara (0 = cámara principal, 1 = cámara secundaria, etc.)
CAMERA_INDEX = 0

# Resolución de la cámara
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ==================== CONFIGURACIÓN DE VISUALIZACIÓN ====================
# Colores en formato BGR
COLOR_WITH_HELMET = (0, 255, 0)      # Verde
COLOR_WITHOUT_HELMET = (0, 0, 255)   # Rojo
COLOR_TEXT = (255, 255, 255)         # Blanco

# Grosor de líneas
BOX_THICKNESS = 2
TEXT_THICKNESS = 2

# Tamaño de fuente
FONT_SCALE = 0.6

# ==================== CONFIGURACIÓN DE MENSAJES ====================
# Plantilla del mensaje de WhatsApp
WHATSAPP_MESSAGE_TEMPLATE = """
🚨 ALERTA DE SEGURIDAD - DETECCIÓN DE CASCO

⚠️ Se ha detectado una persona SIN CASCO DE SEGURIDAD en el área de trabajo.

📅 Fecha: {fecha}
🕐 Hora: {hora}
📍 Ubicación: Área de construcción
🎯 Nivel de confianza: {confianza}%

⚡ ACCIÓN REQUERIDA:
Por favor, verifique el área inmediatamente y asegúrese de que todo el personal use el equipo de protección requerido.

---
Sistema automático de vigilancia de seguridad
SENATI - Ingeniería Civil
""".strip()

# ==================== CONFIGURACIÓN DE LOGS ====================
# Nivel de logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Archivo de logs
LOG_FILE = LOGS_DIR / 'detecciones.log'

# Formato de logs
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

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
    'momentum': 0.937,
    'weight_decay': 0.0005,
}

# ==================== VALIDACIÓN DE CONFIGURACIÓN ====================
def validate_config():
    """Valida que la configuración sea correcta"""
    errors = []
    
    # Validar modelo
    if not MODEL_PATH.exists():
        errors.append(f"⚠️ Modelo no encontrado en: {MODEL_PATH}")
        errors.append("   Descarga o entrena un modelo y colócalo en la carpeta 'models/'")
    
    # Validar Twilio
    if ENABLE_WHATSAPP:
        if 'tu_account_sid_aqui' in TWILIO_ACCOUNT_SID:
            errors.append("⚠️ Configura TWILIO_ACCOUNT_SID en config.py")
        if 'tu_auth_token_aqui' in TWILIO_AUTH_TOKEN:
            errors.append("⚠️ Configura TWILIO_AUTH_TOKEN en config.py")
    
    return errors

# ==================== INFORMACIÓN DEL SISTEMA ====================
def print_config():
    """Imprime la configuración actual"""
    print("=" * 60)
    print("🛡️  CONFIGURACIÓN DEL SISTEMA DE DETECCIÓN DE CASCOS")
    print("=" * 60)
    print(f"\n📁 Directorios:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Modelos: {MODELS_DIR}")
    print(f"   Assets: {ASSETS_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    print(f"\n🤖 Modelo:")
    print(f"   Ruta: {MODEL_PATH}")
    print(f"   Confianza mínima: {CONFIDENCE_THRESHOLD}")
    print(f"   Tamaño de imagen: {IMG_SIZE}")
    print(f"\n📱 WhatsApp/Twilio:")
    print(f"   Habilitado: {ENABLE_WHATSAPP}")
    print(f"   Teléfono destino: {DESTINATION_PHONE}")
    print(f"\n🎥 Cámara:")
    print(f"   Índice: {CAMERA_INDEX}")
    print(f"   Resolución: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"\n⚙️  Detección:")
    print(f"   Umbral de frames: {FRAMES_THRESHOLD}")
    print(f"   Cooldown de alertas: {ALERT_COOLDOWN}s")
    print("=" * 60)
    print()

if __name__ == "__main__":
    print_config()
    errors = validate_config()
    if errors:
        print("\n❌ Errores de configuración:")
        for error in errors:
            print(error)
    else:
        print("✅ Configuración válida")