weights_and_biases = []

def calculate_loss_of_neural_network():
    return 0

# Start

increment = 0.000001
for weight_or_bias in weights_and_biases:
    original_loss = calculate_loss_of_neural_network()
    weight_or_bias += increment
    recalculated_loss = calculate_loss_of_neural_network()
    difference = original_loss - recalculated_loss
    slope = difference / increment
    weight_or_bias -= increment
    weight_or_bias.slope = slope

learning_rate = 0.1
for weight_or_bias in weights_and_biases:
    weight_or_bias -= weight_or_bias.slope * learning_rate
    
