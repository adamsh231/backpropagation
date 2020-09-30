main() {
  double dotProduct(List x, List y){
    double hasil = 0;
    for( var i = 0 ; i < x.length; i++ ) { 
      hasil = hasil + (x[i]*y[i]);
    }
    return hasil;
  }
  
  List weight = [[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
                 [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]];
  List input = [1,2,3,4,5,6,7,8];
  List bias = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5]];
  
  List outcome;
  List hasil = new List();
  hasil.add(input);
   for( var i = 0 ; i < weight.length; i++ ) { 
      outcome = new List(weight[i].length);
      for( var j = 0 ; j < weight[i].length; j++ ) { 
        outcome[j] = dotProduct(input, weight[i][j]) + bias[i][j];
      }
   }
  print(outcome);
}