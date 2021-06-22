import 'dart:io';
import 'Models/dataset.dart';
import 'Models/layer.dart';
import 'Models/neuralnetwork.dart';
import 'Models/neuron.dart';
import 'src/utils.dart' as utils;

var learningRate = 0.01;
var epochs = 100;
var input_n = 4;
var output_n = 1;
var layers_n = 3;
var hidden_layer_n = 8;
var datasetName = "banknote";

void main(List<String> arguments) async {
  Neuron.setRange(0, 1);

  nn = NeuralNetwork([
    Layer([]),
    Layer.hidden(hidden_layer_n, input_n,
        ws: _generateWeights(input_n, hidden_layer_n),
        biasesWeights: _generateBiasWeights(hidden_layer_n),
        bias: _generateBiasValue()),
    Layer.hidden(output_n, hidden_layer_n,
        ws: _generateWeights(hidden_layer_n, output_n),
        biasesWeights: _generateBiasWeights(output_n),
        bias: _generateBiasValue())
  ]);

  var dataset = await loadDataset('train.txt', write: true);

  print('training...');
  train(dataset, epochs, learningRate);

  print('============');
  print('Output after training');
  print('============');
  for (var i in dataset.pairs) {
    forward(i.input_data);
    var out = nn.layers[layers_n - 1].neurons.map((e) => e.value).toList();
    print('output: ' +
        utils.deNormalizeData(out).toString() +
        ' desired = ${i.output_data}');
  }
  print('============');
  print('Predictiong');
  print('============');
  predict();
}

void predict() async {
  var datapredict = await loadDataset('test.txt');
  for (var i in datapredict.pairs) {
    forward(i.input_data);
    var out = nn.layers[layers_n - 1].neurons.map((e) => e.value).toList();
    var desired = utils.deNormalizeData(i.output_data);
    print('predict: ' +
        utils.deNormalizeData(out).toString() +
        ", desired: $desired}");
  }
}

Future<Dataset> loadDataset(filename, {bool write = false}) async {
  var dataset = Dataset([]);
  var data = await File('datasets/$datasetName/$filename').readAsString();
  var allSamples = <double>[];
  for (var rowData in data.split('\n')) {
    if (rowData.isEmpty) continue;
    var sampleInput = <double>[];
    var sampleOutput = <double>[];
    for (var feature in rowData.split(',')) {
      feature.replaceAll(' ', ''); //remove white spaces
      if (feature == '') continue;
      sampleInput.add(double.parse(feature));
    }

    // sampleOutput.addAll(_encodeOutPut(sampleInput.last.toInt()));
    sampleOutput.add(sampleInput.last);
    sampleInput.removeLast();
    allSamples.addAll(sampleInput + sampleOutput);
    dataset.pairs.add(Pair(sampleInput, sampleOutput));
  }
  // utils.minValue = allSamples.reduce(min);
  // utils.maxValue = allSamples.reduce(max);
  // for (var i = 0; i < dataset.pairs.length; i++) {
  //   dataset.pairs[i].input_data =
  //       utils.normalizeData(dataset.pairs[i].input_data);
  //   dataset.pairs[i].output_data =
  //       utils.normalizeData(dataset.pairs[i].output_data);
  // }
  dataset.pairs.shuffle();
  // var toWrite = '';
  // for (var i in dataset.pairs) toWrite += i.output_data[0].toString() + '\n';
  // if (write)
  //   await File('datasets/$datasetName/test/test.txt').writeAsString(toWrite);
  return dataset;
}

List<List<double>> _generateWeights(int fi, int n) {
  List<List<double>> res = [];
  for (int i = 0; i < n; i++) {
    List<double> iRes = [];
    for (int j = 0; j < fi; j++) {
      var r = -2.4 / fi;
      iRes.add(utils.randomWeight((-2.4 / fi), (2.4 / fi)));
    }
    res.add(iRes);
  }
  return res;
}

List<double> _generateBiasWeights(int fi) {
  var res = <double>[];
  for (int i = 0; i < fi; i++) {
    res.add(utils.randomWeight((-2.4 / fi), (2.4 / fi)));
  }
  return res;
}

double _generateBiasValue() => (utils.randomWeight(-1, 1));
