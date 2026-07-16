#include <iostream>
#include <vector>
#include <string>
#include "utec/utils/csv_loader.h"
#include "utec/agent/fraud_agent.h"

int main(int argc, char* argv[]) {
    // Modo de predicción interactiva (Bridge con el Backend)
    if (argc > 1 && std::string(argv[1]) == "predict") {
        if (argc != 33) {
            std::cerr << "Error: Se esperaban 30 features y el path del modelo.\n";
            return 1;
        }
        
        std::string model_path = argv[2];
        utec::agent::FraudAgent agent;
        agent.load_model(model_path);
        
        utec::algebra::Tensor<float, 2> input(std::array<size_t, 2>{1, 30});
        for (int i = 0; i < 30; ++i) {
            float val = std::stof(argv[i + 3]);
            if (i == 0) val = val / 100000.0f; // Normalizar Time
            if (i == 29) val = val / 1000.0f;  // Normalizar Amount
            input(0, i) = val;
        }
        
        int prediction = agent.predict(input);
        std::cout << "{\"prediction\": " << prediction << "}\n";
        return 0;
    }

    // Modo Entrenamiento
    int epochs = 50;
    if (argc > 1 && std::string(argv[1]) == "train") {
        if (argc > 2) epochs = std::stoi(argv[2]);
    } else if (argc > 1) {
        std::cerr << "Uso: ./fraud_net train [epochs] O ./fraud_net predict [model_path] [features...]\n";
        return 1;
    }

    std::string model_path = "../data/model_weights_" + std::to_string(epochs) + ".bin";
    std::cout << "=== Sistema de Entrenamiento (" << epochs << " Epochs) ===\n";
    
    auto dataset = utec::utils::CSVLoader::load_credit_card_data("../data/creditcard.csv", 1000);
    utec::agent::FraudAgent agent;
    float learning_rate = 0.01f;
    agent.train(dataset.features, dataset.labels, epochs, learning_rate);
    
    agent.save_model(model_path);
    std::cout << "Modelo guardado exitosamente en " << model_path << "\n";
    
    agent.evaluate(dataset.features, dataset.labels);
    std::cout << "Proceso finalizado.\n";
    return 0;
}
