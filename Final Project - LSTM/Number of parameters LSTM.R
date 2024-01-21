N <- 1
M <- 10  




LSTM.numb.parameters = 4*( (N+1)*M + M^2 )
Dense.numb.parameters = 1 * ( 50 + 1 )
Total.numb.parameters = LSTM.numb.parameters + Dense.numb.parameters