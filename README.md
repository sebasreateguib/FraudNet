# FraudNet Intelligence: Neural Network Fraud Detection System

FraudNet Intelligence es un proyecto avanzado de Machine Learning y Arquitectura de Software que reemplaza el clásico "Pong AI". Utiliza una Red Neuronal implementada **completamente desde cero en C++** (sin librerías externas de ML) conectada a un entorno Web interactivo de clase empresarial.

## 🧠 Arquitectura del Sistema

El proyecto está compuesto por tres capas principales:

1. **Core de Inferencia (C++)**: La red neuronal (Multi-Layer Perceptron) programada desde cero usando tensores de álgebra lineal propios. Entrenada con Descenso de Gradiente, cuenta con capas densas y funciones de activación Sigmoide. Arquitectura de red: `30 inputs → 16 ocultas → 8 ocultas → 1 output`.
2. **Bridge API (Node.js/Express)**: Un servidor intermedio que recibe peticiones web en JSON, orquesta los llamados de sistema (exec) al binario C++ pre-compilado enviándole vectores matemáticos, y devuelve las predicciones al cliente.
3. **Dashboard de Monitoreo (React + Vite)**: Una interfaz "Minimalista Premium" de altísima calidad que simula un NOC (Network Operations Center) bancario, permitiendo inyectar transacciones en tiempo real.

---

## 📊 Entendiendo el Dataset (Kaggle)

El sistema se alimenta de un famoso dataset real de tarjetas de crédito europeas publicado en Kaggle (284,807 transacciones). 

### 1. Componentes Principales (PCA) y Privacidad
Por temas legales y de privacidad (GDPR), el banco no puede publicar datos sensibles (ej. Ubicación, Comercio, CVV). En su lugar, el dataset fue sometido a un algoritmo llamado **PCA (Principal Component Analysis)**.
* Los atributos originales fueron comprimidos y transformados en **28 variables matemáticas ocultas (`V1` a `V28`)**.
* A la red neuronal no le importa qué significan en la vida real. A través del entrenamiento, la IA es capaz de descubrir los patrones *geométricos* de estas variables en un espacio de 30 dimensiones para trazar un hiperplano que separa compras normales de transacciones fraudulentas.

### 2. Clases Desbalanceadas y Normalización
Originalmente, el dataset tenía un enorme desbalance (sólo 0.17% de fraudes). Si se entrena una IA con esto, sufrirá un colapso y aprenderá a decir "Aprobado" el 100% de las veces. 
* **Solución**: Programamos el `csv_loader.cpp` para extraer un dataset perfectamente balanceado (50% fraudes y 50% normales).
* Además, aplicamos **normalización de entrada** a las columnas de `Time` (dividido entre 100,000) y `Amount` (dividido entre 1,000) para prevenir la *explosión de gradientes* (Gradient Explosion) que corrompía la red.

---

## 🎓 Dinámicas de Aprendizaje (Underfitting vs Convergencia)

Una de las características más interesantes de este proyecto es que podemos presenciar cómo "piensa" y madura la Red Neuronal modificando sus Épocas de Entrenamiento (Epochs). En el Dashboard web puedes cambiar entre tres "cerebros" matemáticos.

- **Modelo de 50 Epochs (Subajustado / Underfitting)**: A las 50 iteraciones, el modelo es altamente paranoico. Logra atrapar casi todos los fraudes, pero sufre de muchísimos **Falsos Positivos** (bloquea a casi 300 tarjetas normales por error).
- **Modelo de 200 Epochs (Estabilizado / Convergencia)**: Al darle más tiempo para ajustar sus pesos (Weights) y sesgos (Bias), la red perfecciona su margen. Reduce los Falsos Positivos a casi **cero**, incrementando el *Accuracy* por encima del 92%.

**Dato Curioso**: Si usas el botón *"Transacción Confusa (Reto IA)"* en la web, se inyecta un vector matemático Fronterizo (Borderline). El modelo de 50 Epochs se asusta y la marca como **FRAUDE** (falso positivo), mientras que el de 200 Epochs sabe reconocer la sutil firma y la **APRUEBA**. ¡El modelo 200 realmente aprendió la *verdadera forma de la variedad (manifold) de fraude* en vez de solo adivinar!

---

## 🚀 Guía de Ejecución

Para iniciar todo el proyecto, necesitas levantar las 3 capas:

### 1. Compilar el Motor C++ (y generar los pesos)
```bash
cd fraud_net
mkdir build
cd build
cmake ..
cmake --build .

# Entrenar y generar los 3 cerebros (esto generará archivos .bin en data/)
./fraud_net train 50
./fraud_net train 100
./fraud_net train 200
```

### 2. Levantar el Bridge API (Node.js)
Abre una nueva terminal.
```bash
cd backend
npm install
node server.js
# Deberías ver: "Backend Puente escuchando en http://localhost:3001"
```

### 3. Levantar el Frontend (React)
Abre una tercera terminal.
```bash
cd frontend
npm install
npm run dev
# Accede a http://localhost:5173 en tu navegador
```

### 🛠 Consideraciones Técnicas para la Demostración
- Asegúrate siempre de detener (`Ctrl + C`) y reiniciar el `node server.js` si haces cambios en el código C++, ya que el backend maneja las rutas hacia el binario pre-compilado.
- Si un puerto está en uso, verifica qué terminal está corriendo procesos en el puerto 3001 o 5173.
