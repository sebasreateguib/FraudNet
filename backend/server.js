const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const BINARY_PATH = path.join(__dirname, '../fraud_net/build/fraud_net');

app.post('/api/predict', (req, res) => {
    const { features, epochs = 50 } = req.body;
    
    if (!features || features.length !== 30) {
        return res.status(400).json({ error: 'Se esperaban exactamente 30 variables' });
    }

    // Unir los 30 features separados por espacio
    const argsString = features.join(' ');
    // El binario espera el path del modelo antes de las features
    const modelPath = path.join(__dirname, `../fraud_net/data/model_weights_${epochs}.bin`);
    const command = `"${BINARY_PATH}" predict "${modelPath}" ${argsString}`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error('Error executando binario de C++:', stderr || error.message);
            return res.status(500).json({ error: 'Fallo interno del modelo C++' });
        }
        try {
            // Se espera que el C++ imprima {"prediction": 0} o {"prediction": 1}
            const result = JSON.parse(stdout.trim().split('\n').pop());
            res.json(result);
        } catch (e) {
            console.error('No se pudo parsear el JSON:', stdout);
            res.status(500).json({ error: 'Respuesta inválida desde el Core de IA' });
        }
    });
});

const PORT = 3001;
app.listen(PORT, () => console.log(`Backend Puente escuchando en http://localhost:${PORT}`));
