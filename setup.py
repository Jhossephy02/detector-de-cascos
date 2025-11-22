# setup.py
"""
Script de configuración automática del Sistema de Detección de Cascos
Versión Mejorada con más validaciones y opciones
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess
import platform

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "="*80)
    print(f"  {text}".center(80))
    print("="*80 + "\n")

def print_step(step_num, total_steps, description):
    """Imprime el paso actual"""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 80)

def check_python_version():
    """Verifica la versión de Python"""
    print("🔍 Verificando versión de Python...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detectado")
        print("   Se requiere Python 3.8 o superior")
        print("\n📥 Descarga Python desde: https://www.python.org/downloads/")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    # Información del sistema
    print(f"   Sistema: {platform.system()} {platform.release()}")
    print(f"   Arquitectura: {platform.machine()}")
    
    return True

def create_directories():
    """Crea las carpetas necesarias"""
    print("📁 Creando estructura de directorios...")
    
    directories = {
        'models': 'Modelos entrenados',
        'assets': 'Recursos (audio, imágenes)',
        'logs': 'Archivos de log',
        'data': 'Datos del sistema',
        'data/detections': 'Capturas de detecciones',
    }
    
    for directory, description in directories.items():
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"   ✅ Creado: {directory:<20} ({description})")
        else:
            print(f"   ⏭️  Existe: {directory:<20} ({description})")
    
    return True

def install_dependencies():
    """Instala las dependencias necesarias"""
    print("📦 Instalando dependencias...")
    
    if not Path('requirements.txt').exists():
        print("   ❌ No se encontró requirements.txt")
        return False
    
    try:
        # Leer requirements
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"   📋 {len(requirements)} paquetes por instalar")
        print("   ⏳ Esto puede tomar varios minutos...")
        
        # Actualizar pip
        print("\n   Actualizando pip...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            capture_output=True,
            check=True
        )
        
        # Instalar dependencias
        print("   Instalando paquetes...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✅ Dependencias instaladas correctamente")
            
            # Mostrar paquetes instalados
            print("\n   📊 Verificando instalación...")
            key_packages = ['ultralytics', 'opencv-python', 'torch', 'pygame', 'twilio']
            
            for package in key_packages:
                try:
                    __import__(package.replace('-', '_'))
                    print(f"      ✅ {package}")
                except ImportError:
                    print(f"      ❌ {package} - No instalado")
            
            return True
        else:
            print(f"   ❌ Error instalando dependencias:")
            print(f"   {result.stderr[:500]}")
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
        
        # Validar que sea un modelo válido
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            print(f"   ✅ Modelo válido")
            print(f"   📊 Clases: {list(model.names.values())}")
            return True
        except Exception as e:
            print(f"   ⚠️  Modelo corrupto o inválido: {e}")
            return False
    else:
        print("   ⚠️  Modelo NO encontrado")
        print("\n   📝 OPCIONES PARA OBTENER EL MODELO:")
        print("")
        print("   1️⃣  Entrenar en Google Colab (RECOMENDADO)")
        print("       • Abre train_colab_complete.py en Colab")
        print("       • Ejecuta todas las celdas")
        print("       • Descarga best.pt")
        print("       • Colócalo en: models/best.pt")
        print("")
        print("   2️⃣  Descargar modelo pre-entrenado")
        print("       • Busca en Roboflow Universe")
        print("       • Tema: 'Safety Helmet Detection'")
        print("       • Formato: YOLOv8")
        print("       • Colócalo en: models/best.pt")
        print("")
        print("   3️⃣  Usar modelo genérico (menos preciso)")
        print("       • El sistema descargará yolov8n.pt")
        print("       • No está entrenado para cascos")
        print("")
        return False

def create_env_file():
    """Crea archivo .env desde .env.example"""
    print("📝 Configurando variables de entorno...")
    
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_file.exists():
        print("   ⏭️  Archivo .env ya existe")
        return True
    
    if not env_example.exists():
        print("   ⚠️  No se encontró .env.example")
        return False
    
    try:
        shutil.copy(env_example, env_file)
        print("   ✅ Archivo .env creado desde .env.example")
        print("   📝 Edita .env para personalizar configuración")
        return True
    except Exception as e:
        print(f"   ❌ Error creando .env: {e}")
        return False

def setup_twilio():
    """Guía para configurar Twilio"""
    print("📱 Configuración de WhatsApp/Twilio...")
    
    print("\n   ¿Deseas configurar notificaciones por WhatsApp ahora? (s/n): ", end='')
    try:
        response = input().strip().lower()
    except KeyboardInterrupt:
        print("\n   ⏭️  Omitido")
        return True
    
    if response == 's':
        print("\n   📝 PASOS PARA CONFIGURAR TWILIO:")
        print("   " + "="*76)
        print("   1. Crea cuenta gratuita:")
        print("      https://www.twilio.com/try-twilio")
        print("")
        print("   2. En el Dashboard de Twilio, copia:")
        print("      • Account SID")
        print("      • Auth Token")
        print("")
        print("   3. Configura WhatsApp Sandbox:")
        print("      • Console → Messaging → Try it out → WhatsApp")
        print("      • Envía el código desde tu WhatsApp al número de Twilio")
        print("")
        print("   4. Edita el archivo .env con tus credenciales:")
        print("      TWILIO_ACCOUNT_SID=tu_account_sid")
        print("      TWILIO_AUTH_TOKEN=tu_auth_token")
        print("      DESTINATION_PHONE=+51xxxxxxxxx")
        print("      ENABLE_WHATSAPP=true")
        print("   " + "="*76)
        
        print("\n   Presiona Enter cuando hayas completado la configuración...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n   ⏭️  Configuración pendiente")
            return True
        
        # Verificar configuración
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            account_sid = os.getenv('TWILIO_ACCOUNT_SID', '')
            
            if account_sid and 'tu_account_sid' not in account_sid:
                print("   ✅ Credenciales configuradas en .env")
                return True
            else:
                print("   ⚠️  Credenciales aún no configuradas")
                print("   💡 Puedes configurarlas más tarde editando .env")
                return False
        except:
            print("   ⚠️  No se pudo verificar .env")
            return False
    else:
        print("   ⏭️  WhatsApp omitido")
        print("   💡 Para habilitar después: ENABLE_WHATSAPP=true en .env")
        return True

def test_camera():
    """Verifica que la cámara funcione"""
    print("🎥 Probando cámara...")
    
    try:
        import cv2
        
        print("   Intentando abrir cámara...")
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                h, w = frame.shape[:2]
                print(f"   ✅ Cámara funcionando")
                print(f"   📐 Resolución: {w}x{h}")
                cap.release()
                return True
            else:
                print("   ⚠️  Cámara abierta pero no captura frames")
                cap.release()
                return False
        else:
            print("   ❌ No se pudo abrir la cámara")
            print("\n   🔧 SOLUCIONES:")
            print("      • Verifica que la cámara esté conectada")
            print("      • Cierra otras apps que usen la cámara")
            print("      • Prueba CAMERA_INDEX=1 en .env")
            print("      • En Windows: Verifica permisos en Configuración")
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

def test_imports():
    """Prueba que todos los imports funcionen"""
    print("🧪 Probando imports de módulos...")
    
    modules_to_test = {
        'cv2': 'OpenCV',
        'torch': 'PyTorch',
        'ultralytics': 'YOLOv8',
        'pygame': 'Pygame (audio)',
        'twilio': 'Twilio',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
    }
    
    failed = []
    
    for module, name in modules_to_test.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n   ⚠️  {len(failed)} módulos faltantes")
        print("   Ejecuta: pip install -r requirements.txt")
        return False
    
    return True

def create_readme():
    """Crea README con instrucciones"""
    print("📄 Creando README.md...")
    
    readme_content = """# Sistema de Detección de Cascos de Seguridad

Sistema inteligente de detección de cascos usando YOLOv8 con alertas por WhatsApp.

## 🚀 Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar (ejecutar una vez)
python setup.py

# 3. Entrenar modelo en Colab
# Abre train_colab_complete.py en Google Colab

# 4. Ejecutar sistema
python main.py
```

## ⚙️ Configuración

Edita `.env` para personalizar:
- Umbrales de detección
- Credenciales de Twilio
- Configuración de cámara
- Parámetros de alertas

## 🎮 Controles

- `q` - Salir
- `r` - Resetear alertas
- `s` - Guardar screenshot
- `p` - Pausar/Reanudar
- `d` - Toggle debug

## 📝 Archivos Principales

- `main.py` - Sistema principal
- `config.py` - Configuración
- `setup.py` - Script de configuración
- `test_model.py` - Pruebas del modelo
- `.env` - Variables de entorno

## 🆘 Soporte

Para problemas, revisa los logs en `logs/detecciones.log`
"""
    
    try:
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("   ✅ README.md creado")
        return True
    except Exception as e:
        print(f"   ❌ Error creando README: {e}")
        return False

def show_next_steps(model_exists):
    """Muestra los próximos pasos"""
    print_header("🎯 PRÓXIMOS PASOS")
    
    if not model_exists:
        print("📌 PRIORIDAD: OBTENER MODELO")
        print("=" * 80)
        print("")
        print("1️⃣  Entrenar en Google Colab (RECOMENDADO):")
        print("    • Abre Google Colab: https://colab.research.google.com")
        print("    • Sube el archivo: train_colab_complete.py")
        print("    • Ejecuta todas las celdas (1-3 horas)")
        print("    • Descarga best.pt")
        print("    • Colócalo en: models/best.pt")
        print("")
        print("2️⃣  Configurar sistema:")
        print("    • Edita .env con tus preferencias")
        print("    • (Opcional) Configura Twilio para WhatsApp")
        print("")
        print("3️⃣  Probar modelo:")
        print("    python test_model.py")
        print("")
        print("4️⃣  Ejecutar sistema:")
        print("    python main.py")
    else:
        print("¡Sistema listo para usar!")
        print("=" * 80)
        print("")
        print("1️⃣  Probar modelo:")
        print("    python test_model.py")
        print("")
        print("2️⃣  Configurar alertas (opcional):")
        print("    • Edita .env con credenciales de Twilio")
        print("    • ENABLE_WHATSAPP=true")
        print("")
        print("3️⃣  Ejecutar sistema:")
        print("    python main.py")
        print("")
        print("4️⃣  Controles durante ejecución:")
        print("    • q: Salir")
        print("    • r: Resetear alertas")
        print("    • s: Guardar screenshot")
        print("    • p: Pausar/Reanudar")
    
    print("")
    print("=" * 80)

def main():
    """Función principal de configuración"""
    print_header("🛡️ CONFIGURADOR DEL SISTEMA DE DETECCIÓN DE CASCOS")
    
    print("Este script configurará automáticamente tu entorno.")
    print("Tiempo estimado: 5-10 minutos")
    print("")
    print("Presiona Enter para comenzar o Ctrl+C para cancelar...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Configuración cancelada")
        return
    
    # Pasos de configuración
    steps = [
        ("Versión de Python", check_python_version),
        ("Crear directorios", create_directories),
        ("Crear archivo .env", create_env_file),
        ("Instalar dependencias", install_dependencies),
        ("Probar imports", test_imports),
        ("Verificar modelo", check_model),
        ("Configurar Twilio", setup_twilio),
        ("Probar cámara", test_camera),
        ("Crear audio de alerta", create_test_audio),
        ("Crear README", create_readme),
    ]
    
    results = []
    total_steps = len(steps)
    model_exists = False
    
    for i, (step_name, step_func) in enumerate(steps, 1):
        print_step(i, total_steps, step_name)
        
        try:
            result = step_func()
            results.append((step_name, result))
            
            if step_name == "Verificar modelo":
                model_exists = result
            
            if not result and step_name in ["Versión de Python", "Instalar dependencias"]:
                print(f"\n❌ Error crítico en: {step_name}")
                print("No se puede continuar sin resolver este problema.")
                return
                
        except Exception as e:
            print(f"❌ Error en {step_name}: {e}")
            results.append((step_name, False))
    
    # Resumen final
    print_header("📊 RESUMEN DE CONFIGURACIÓN")
    
    critical_steps = ["Versión de Python", "Instalar dependencias", "Probar imports"]
    important_steps = ["Verificar modelo", "Probar cámara"]
    
    for step_name, result in results:
        if step_name in critical_steps:
            status = "✅" if result else "❌"
        elif step_name in important_steps:
            status = "✅" if result else "⚠️"
        else:
            status = "✅" if result else "⏭️"
        
        print(f"{status} {step_name}")
    
    # Contar éxitos
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print("")
    print(f"Completado: {success_count}/{total_count} pasos")
    print("")
    
    if success_count == total_count:
        print("🎉 ¡Configuración 100% completa!")
    elif success_count >= total_count - 2:
        print("✅ Configuración casi completa")
        print("Solo faltan algunos pasos opcionales.")
    else:
        print("⚠️  Configuración incompleta")
        print("Revisa los errores anteriores.")
    
    # Mostrar próximos pasos
    show_next_steps(model_exists)
    
    print_header("✨ CONFIGURACIÓN FINALIZADA")
    print("💡 Para más ayuda, lee README.md")
    print("")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Configuración interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)