from pdearena.pl_models.cnnmodel import (
    CNNModel, 
    NeuralSolver, 
    NeuralCFL,
    ShockCast
)

MODELS = {
    "cnn": CNNModel,
    "neural_solver": NeuralSolver,
    "neural_cfl": NeuralCFL,
    "shock_cast": ShockCast,
}