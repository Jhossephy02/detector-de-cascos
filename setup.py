# setup.py
"""
Configurador automático del Sistema de Detección de Cascos
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path


def header(text):
    print("\n" + "=" * 70)
    print(f"  {text}".center(70))
    print("=" * 70 + "\n")


def step(num, total, desc):
    print(f"\n[{num}/{total}] {desc}")
    print("-" * 70)


def check_python():
    """Verifica versión de Python"""
    print("🔍 Verificando Python...")
    v = sys.version_info
    
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print(f"❌ Python {v.major}.{v.minor} - Se requiere 3.8+")
        print("   Descarga desde: python.org/downloads")
        return False
    
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")
    print(f"   Sistema: {platform.system()} {platform.machine()}")
    return True


def create_dirs():
    """Crea estructura de carpetas"""
    print("📁 Creando directorios...")
    
    dirs = {
        'models': 'Modelos .pt',
        'assets': 'Audio y recursos',
        'logs': 'Archivos de log',
        'data': 'Datos del sistema',
        'data/detections': 'Capturas',
    }
    
    for d, desc in dirs.items():
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"   ✅ Creado: {d}")
        else:
            print(f"   ⏭️  Existe: {d}")
    return True


def install_deps():
    """Instala dependencias"""
    print("📦 Instalando dependencias...")
    
    req_file = Path('requirements.txt')
    if not req_file.exists():
        print("   ❌ requirements.txt no encontrado")
        return False
    
    print("   ⏳ Esto puede tomar varios minutos...")
    
    try:
        # Actualizar pip
        subprocess.run([sys.executable, '-m', 'pip', 'install', 
                       '--upgrade', 'pip'], capture_output=True, check=True)
        
        # Instalar requirements
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("   ✅ Dependencias instaladas")
            return True
        else:
            print(f"   ❌ Error: {result.stderr[:300]}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_imports():
    """Prueba imports críticos"""
    print("🧪 Verificando módulos...")
    
    modules = {
        'cv2': 'OpenCV',
        'torch': 'PyTorch',
        'ultralytics': 'YOLOv8',
        'pygame': 'Audio',
        'numpy': 'NumPy',
    }
    
    failed = []
    for mod, name in modules.items():
        try:
            __import__(mod)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            failed.append(mod)
    
    return len(failed) == 0


def check_model():
    """Verifica el modelo"""
    print("🤖 Verificando modelo...")
    
    model_path = Path('models/best.pt')
    
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Modelo encontrado ({size:.1f} MB)")
        
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            print(f"   ✅ Modelo válido")
            print(f"   📊 Clases: {list(model.names.values())}")
            return True
        except Exception as e:
            print(f"   ⚠️  Modelo inválido: {e}")
            return False
    else:
        print("   ⚠️  Modelo NO encontrado")
        print("""
   📝 OPCIONES:
   
   1️⃣  Entrenar en Google Colab (RECOMENDADO)
       • Sube train_colab.py a colab.google.com
       • Activa GPU y ejecuta todas las celdas
       • Descarga best.pt → models/
   
   2️⃣  Descargar de Roboflow Universe
       • Busca 'Safety Helmet Detection'
       • Descarga formato YOLOv8
       • Coloca en models/best.pt
""")
        return False


def create_env():
    """Crea archivo .env"""
    print("📝 Configurando .env...")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_file.exists():
        print("   ⏭️  .env ya existe")
        return True
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("   ✅ .env creado desde .env.example")
        return True
    else:
        # Crear .env básico
        content = """# Configuración básica
CONFIDENCE_THRESHOLD=0.5
ENABLE_WHATSAPP=false
ENABLE_AUDIO=true
CAMERA_INDEX=0
"""
        env_file.write_text(content)
        print("   ✅ .env básico creado")
        return True


def test_camera():
    """Prueba la cámara"""
    print("🎥 Probando cámara...")
    
    try:
        import cv2
        
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"   ✅ Cámara {idx}: {w}x{h}")
                    cap.release()
                    return True
                cap.release()
        
        print("   ❌ No se encontró cámara")
        print("   🔧 Verifica conexión o cambia CAMERA_INDEX en .env")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def create_audio():
    """Crea archivo de audio"""
    print("🔊 Creando audio de alerta...")
    
    try:
        import numpy as np
        from scipy.io import wavfile
        
        sr = 44100
        duration = 1.5
        freq = 880
        
        t = np.linspace(0, duration, int(sr * duration))
        envelope = np.exp(-3 * t)
        audio = np.sin(2 * np.pi * freq * t) * envelope * 0.3
        audio = (audio * 32767).astype(np.int16)
        
        Path('assets').mkdir(exist_ok=True)
        wavfile.write('assets/alert.wav', sr, audio)
        print("   ✅ alert.wav creado")
        return True
    except Exception as e:
        print(f"   ⚠️  No se pudo crear: {e}")
        print("   Se creará automáticamente al ejecutar")
        return True


def main():
    """Función principal"""
    header("🛡️ CONFIGURADOR - DETECCIÓN DE CASCOS")
    
    print("Este script configurará tu entorno automáticamente.")
    print("Tiempo estimado: 3-5 minutos\n")
    print("Presiona Enter para comenzar...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Cancelado")
        return
    
    # Pasos
    steps = [
        ("Python", check_python),
        ("Directorios", create_dirs),
        ("Archivo .env", create_env),
        ("Dependencias", install_deps),
        ("Imports", test_imports),
        ("Modelo", check_model),
        ("Cámara", test_camera),
        ("Audio", create_audio),
    ]
    
    results = []
    model_ok = False
    
    for i, (name, func) in enumerate(steps, 1):
        step(i, len(steps), name)
        try:
            result = func()
            results.append((name, result))
            if name == "Modelo":
                model_ok = result
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((name, False))
    
    # Resumen
    header("📊 RESUMEN")
    
    critical = ["Python", "Dependencias", "Imports"]
    
    for name, result in results:
        if name in critical:
            icon = "✅" if result else "❌"
        else:
            icon = "✅" if result else "⚠️"
        print(f"   {icon} {name}")
    
    success = sum(1 for _, r in results if r)
    print(f"\n   Completado: {success}/{len(results)}")
    
    # Próximos pasos
    header("🎯 PRÓXIMOS PASOS")
    
    if not model_ok:
        print("""
1️⃣  OBTENER MODELO (PRIORITARIO):
    • Abre colab.research.google.com
    • Sube train_colab.py
    • Runtime → Change runtime → GPU
    • Ejecuta todas las celdas
    • Descarga best.pt → models/

2️⃣  Probar modelo:
    python test_model.py

3️⃣  Ejecutar sistema:
    python main.py
""")
    else:
        print("""
1️⃣  Probar modelo:
    python test_model.py

2️⃣  Ejecutar sistema:
    python main.py

3️⃣  (Opcional) Configurar WhatsApp:
    Edita .env con tus credenciales de Twilio
""")
    
    print("=" * 70)
    print("💡 Para más ayuda, lee README.md")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()