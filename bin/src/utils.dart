import 'dart:math' as math;

var maxValue = 0.0;
var minValue = 0.0;
// Random double between min - max
double randomWeight(double min, double max) {
  // return 0.0;
  math.Random rand = new math.Random();
  var res = rand.nextDouble() * (max - min) + min;
  return res;
}

// Sigmoid Function
double sigmoid(double x) {
  var res = (1 / (1 + math.pow(math.e, -1 * x))).toDouble();
  // print(res);
  return res;
}

// relu Function
double relu(double x) {
  print(x);
  return x > 0.0 ? x : 0.0;
}

// tanh Function
double tanh(double x) {
  return (_exp(x) - _exp(-x)) / (_exp(x) + _exp(-x));
}

double _exp(x) => math.pow(math.e, -1 * x);
// threshold Function
double threshold(value, threshold) {
  return value > threshold ? 1 : 0;
}

// Used for Backpropagation
double squaredError(double output, double target) {
  return (0.5 * math.pow(2, (target - output))).toDouble();
}

// Used to calculate the overall error rate
double sumSquaredError(List<double> outputs, List<double> targets) {
  double sum = 0;
  for (int i = 0; i < outputs.length; i++) {
    sum += squaredError(outputs[i], targets[i]);
  }
  return sum;
}

List<double> normalizeData(List<double> data) {
  // if (maxValue == minValue) return data;
  // for (var i = 0; i < data.length; i++) {
  //   var res = (data[i] - minValue) / (maxValue - minValue);
  //   data[i] = res;
  // }
  return data;
}

List<double> deNormalizeData(List<double> data) {
  // if (maxValue == minValue) return data;
  // for (var i = 0; i < data.length; i++) {
  //   var res = (data[i] * (maxValue - minValue)) + minValue;
  //   data[i] = res;
  // }
  return data;
}
