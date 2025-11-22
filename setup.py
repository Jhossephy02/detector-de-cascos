# setup.py
"""
Script de configuración automática del Sistema de Detección de Cascos
Ejecuta este script la primera vez para configurar todo automáticamente
"""

import os
import sys
from pathlib import Path
import subprocess

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Verifica la versión de Python"""
    print("🔍 Verificando versión de Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detectado")
        print("   Se requiere Python 3.8 o superior")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
    return True

def create_directories():
    """Crea las carpetas necesarias"""
    print("📁 Creando estructura de directorios...")
    
    directories = ['models', 'assets', 'logs']
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"   ✅ Creado: {directory}/")
        else:
            print(f"   ⏭️  Ya existe: {directory}/")
    
    return True

def install_dependencies():
    """Instala las dependencias necesarias"""
    print("📦 Instalando dependencias...")
    
    try:
        # Leer requirements.txt
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        print(f"   Instalando {len(requirements)} paquetes...")
        
        # Instalar con pip
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✅ Dependencias instaladas correctamente")
            return True
        else:
            print(f"   ❌ Error instalando dependencias:")
            print(f"   {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("   ❌ No se encontró requirements.txt")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_model():
    """Verifica si existe el modelo"""
    print("🤖 Verificando modelo YOLOv8...")
    
    model_path = Path('models/best.pt')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Modelo encontrado ({size_mb:.1f} MB)")
        return True
    else:
        print("   ⚠️  Modelo NO encontrado")
        print("   ")
        print("   Para obtener el modelo, tienes 3 opciones:")
        print("   ")
        print("   1️⃣  Entrenar en Google Colab (Recomendado)")
        print("       - Abre el notebook proporcionado en Colab")
        print("       - Ejecuta todas las celdas")
        print("       - Descarga best.pt y colócalo en models/")
        print("   ")
        print("   2️⃣  Descargar modelo pre-entrenado")
        print("       - Busca en Roboflow Universe: 'Safety Helmet Detection'")
        print("       - Descarga el modelo en formato YOLOv8")
        print("       - Colócalo en models/best.pt")
        print("   ")
        print("   3️⃣  Entrenar localmente")
        print("       - Prepara tu dataset")
        print("       - Ejecuta el entrenamiento con YOLOv8")
        print("   ")
        return False

def setup_twilio():
    """Guía para configurar Twilio"""
    print("📱 Configuración de WhatsApp/Twilio...")
    
    print("   ¿Deseas configurar notificaciones por WhatsApp? (s/n): ", end='')
    response = input().strip().lower()
    
    if response == 's':
        print("\n   📝 Pasos para configurar Twilio:")
        print("   ")
        print("   1. Crea cuenta gratuita: https://www.twilio.com/try-twilio")
        print("   2. En el Dashboard, copia:")
        print("      - Account SID")
        print("      - Auth Token")
        print("   3. Configura WhatsApp Sandbox:")
        print("      - Console → Messaging → WhatsApp")
        print("      - Envía el código de unión desde tu WhatsApp")
        print("   4. Abre config.py y actualiza:")
        print("      - TWILIO_ACCOUNT_SID")
        print("      - TWILIO_AUTH_TOKEN")
        print("      - DESTINATION_PHONE (tu número con +51)")
        print("   ")
        
        print("   Presiona Enter cuando hayas completado la configuración...")
        input()
        
        # Verificar si se actualizó config.py
        try:
            from config import TWILIO_ACCOUNT_SID, ENABLE_WHATSAPP
            
            if 'tu_account_sid_aqui' not in TWILIO_ACCOUNT_SID:
                print("   ✅ Credenciales de Twilio configuradas")
                return True
            else:
                print("   ⚠️  Credenciales aún no configuradas")
                print("   Puedes configurarlas más tarde en config.py")
                return False
        except:
            print("   ⚠️  No se pudo verificar config.py")
            return False
    else:
        print("   ⏭️  WhatsApp omitido (puedes habilitarlo después en config.py)")
        print("   💡 Tip: Cambia ENABLE_WHATSAPP = False en config.py")
        return True

def test_camera():
    """Verifica que la cámara funcione"""
    print("🎥 Probando cámara...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("   ✅ Cámara funcionando correctamente")
                return True
            else:
                print("   ⚠️  Cámara abierta pero no puede capturar frames")
                return False
        else:
            print("   ❌ No se pudo abrir la cámara")
            print("   ")
            print("   Soluciones:")
            print("   - Verifica que la cámara esté conectada")
            print("   - En config.py, prueba CAMERA_INDEX = 1 o 2")
            print("   - Asegúrate de que otra aplicación no esté usando la cámara")
            return False
            
    except Exception as e:
        print(f"   ❌ Error probando cámara: {e}")
        return False

def create_test_audio():
    """Crea un archivo de audio de prueba"""
    print("🔊 Creando archivo de alerta de audio...")
    
    try:
        import numpy as np
        from scipy.io import wavfile
        
        sample_rate = 44100
        duration = 1.5
        frequency = 880
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        envelope = np.exp(-3 * t)
        audio_data = np.sin(2 * np.pi * frequency * t) * envelope * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        wavfile.write('assets/alert.wav', sample_rate, audio_data)
        print("   ✅ Archivo alert.wav creado")
        return True
        
    except Exception as e:
        print(f"   ⚠️  No se pudo crear archivo de audio: {e}")
        print("   El sistema lo creará automáticamente al ejecutarse")
        return True

def show_next_steps():
    """Muestra los próximos pasos"""
    print_header("🎯 PRÓXIMOS PASOS")
    
    print("Para usar el sistema:")
    print("")
    print("1️⃣  Si aún no tienes el modelo:")
    print("    - Abre el notebook en Google Colab")
    print("    - Entrena el modelo (1-3 horas)")
    print("    - Descarga best.pt a la carpeta models/")
    print("")
    print("2️⃣  Configura Twilio (opcional):")
    print("    - Edita config.py con tus credenciales")
    print("    - O desactiva: ENABLE_WHATSAPP = False")
    print("")
    print("3️⃣  Ejecuta el sistema:")
    print("    python main.py")
    print("")
    print("4️⃣  Controles:")
    print("    - Presiona 'q' para salir")
    print("    - Presiona 'r' para resetear alertas")
    print("")

def main():
    """Función principal de configuración"""
    print_header("🛡️ CONFIGURADOR DEL SISTEMA DE DETECCIÓN DE CASCOS")
    
    print("Este script configurará automáticamente tu entorno.")
    print("Presiona Enter para comenzar o Ctrl+C para cancelar...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Configuración cancelada")
        return
    
    # Ejecutar pasos de configuración
    steps = [
        ("Versión de Python", check_python_version),
        ("Crear directorios", create_directories),
        ("Instalar dependencias", install_dependencies),
        ("Verificar modelo", check_model),
        ("Configurar Twilio", setup_twilio),
        ("Probar cámara", test_camera),
        ("Crear audio de alerta", create_test_audio),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print_header(f"Paso: {step_name}")
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"❌ Error en {step_name}: {e}")
            results.append((step_name, False))
    
    # Resumen final
    print_header("📊 RESUMEN DE CONFIGURACIÓN")
    
    for step_name, result in results:
        status = "✅" if result else "⚠️"
        print(f"{status} {step_name}")
    
    # Contar éxitos
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print("")
    print(f"Completado: {success_count}/{total_count} pasos")
    
    if success_count == total_count:
        print("\n🎉 ¡Configuración completa!")
        print("El sistema está listo para usarse.")
    elif success_count >= total_count - 2:
        print("\n✅ Configuración casi completa")
        print("Solo faltan algunos pasos opcionales.")
    else:
        print("\n⚠️  Configuración incompleta")
        print("Revisa los errores anteriores.")
    
    # Mostrar próximos pasos
    show_next_steps()
    
    print_header("✨ CONFIGURACIÓN FINALIZADA")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Configuración interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        sys.exit(1)