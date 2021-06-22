import 'dataset.dart';
import 'layer.dart';
import 'neuron.dart';
import '../src/utils.dart' as utils;
import 'dart:math' as math;

const loss = 0.001;
double sumSqError = 0.0;
NeuralNetwork nn;

class NeuralNetwork {
  List<Layer> layers = [];
  int size; //# of layers

  NeuralNetwork(List<Layer> l) {
    this.size = l.length;
    this.layers = l;
  }

  int getSize() {
    return this.size;
  }
}

void forward(List<double> inputs, {double desired}) {
  // Bring the inputs into the input layer
  nn.layers[0] = Layer(inputs);

  // Forward propagation
  for (var i = 1; i < nn.layers.length; i++) {
    //all layers
    for (var j = 0; j < nn.layers[i].neurons.length; j++) {
      //all neurons in the current layer
      double sum = 0;
      for (int k = 0; k < nn.layers[i - 1].neurons.length; k++) {
        //all neurons in the prev layer
        sum += nn.layers[i - 1].neurons[k].value * //value of the prev neuron
            nn.layers[i].neurons[j]
                .weights_old[k]; //the weight bettween me (j) and the prev (k)
      }
      sum += nn.layers[i].neurons[j].bias * nn.layers[i].neurons[j].biasWeight;
      var output = utils.sigmoid(sum);
      nn.layers[i].neurons[j].value = output;
      if (desired != null)
        sumSqError += math.pow((desired - output), 2).toDouble();
    }
  }
}

void backward(double learning_rate, Pair datas) {
  int number_layers = nn.layers.length;
  int output_layer_index = number_layers - 1;

  // Update the output layers
  for (int i = 0; i < nn.layers[output_layer_index].neurons.length; i++) {
    //the neurons at the output layer
    double output = nn.layers[output_layer_index].neurons[i].value;
    double target = datas.output_data[i];
    double e = target - output;
    double delta = e * (output * (1 - output));
    nn.layers[output_layer_index].neurons[i].gradient = delta;

    for (int j = 0;
        j < nn.layers[output_layer_index].neurons[i].weights.length;
        j++) {
      // and for each of their weights
      double previous_output =
          nn.layers[output_layer_index - 1].neurons[j].value;
      double error = delta * previous_output;
      double oldW = nn.layers[output_layer_index].neurons[i].weights_old[j];
      nn.layers[output_layer_index].neurons[i].weights[j] =
          oldW + learning_rate * error;
      //update bias weight
      nn.layers[output_layer_index].neurons[i].biasWeight = nn
              .layers[output_layer_index].neurons[i].biasWeight +
          learning_rate * nn.layers[output_layer_index].neurons[i].bias * delta;
    }
  }

  // Update all the subsequent hidden layers
  for (int i = output_layer_index - 1; i > 0; i--) {
    // Backward
    for (int j = 0; j < nn.layers[i].neurons.length; j++) {
      // For all neurons in that layers
      double output = nn.layers[i].neurons[j].value;
      double gradient_sum = sumGradient(nn, j, i + 1);
      double delta = (gradient_sum) * (output * (1 - output));
      nn.layers[i].neurons[j].gradient = delta;

      for (int k = 0; k < nn.layers[i].neurons[j].weights.length; k++) {
        // And for all their weights
        double previous_output = nn.layers[i - 1].neurons[k].value;
        double error = delta * previous_output;
        nn.layers[i].neurons[j].weights[k] =
            nn.layers[i].neurons[j].weights_old[k] + learning_rate * error;
      }
      //update bias weight
      nn.layers[i].neurons[j].biasWeight = nn.layers[i].neurons[j].biasWeight +
          learning_rate * nn.layers[i].neurons[j].bias * delta;
    }
  }

  // Update all the weights
  for (int i = 0; i < nn.layers.length; i++) {
    for (int j = 0; j < nn.layers[i].neurons.length; j++) {
      nn.layers[i].neurons[j].updateWeights();
    }
  }
}

// This function sums up all the gradient connecting a given neuron in a given layer
double sumGradient(NeuralNetwork nn, int n_index, int l_index) {
  double gradient_sum = 0;
  Layer current_layer = nn.layers[l_index];
  for (int i = 0; i < current_layer.neurons.length; i++) {
    Neuron current_neuron = current_layer.neurons[i];
    gradient_sum +=
        current_neuron.weights_old[n_index] * current_neuron.gradient;
  }
  return gradient_sum;
}

// This function is used to train
void train(Dataset dt, int epochs, double learning_rate) {
  epochsLoop:
  for (int i = 0; i < epochs; i++) {
    for (int j = 0; j < dt.pairs.length; j++) {
      forward(dt.pairs[j].input_data, desired: dt.pairs[j].output_data[0]);
      if (sumSqError <= loss) {
        print('breaked at epoch ${i + 1}');
        break epochsLoop;
      }
      backward(learning_rate, dt.pairs[j]);
    }
  }
}
