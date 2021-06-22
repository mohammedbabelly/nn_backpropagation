import 'neuron.dart';

class Layer {
  List<Neuron> neurons = [];

  // Constructor for input layers
  Layer(List<double> inputs) {
    for (int i = 0; i < inputs.length; i++) {
      this.neurons.add(Neuron(inputs[i]));
    }
  }

  // Constructor for hidden & output layers
  Layer.hidden(int number_of_neurons, int weights_per_neuron,
      {List<List<double>> ws, double bias = -1, List<double> biasesWeights}) {
    for (int i = 0; i < number_of_neurons; i++) {
      if (ws != null) {
        this.neurons.add(Neuron.hidden(ws[i], bias, biasesWeights[i]));
      }
    }
  }
}
