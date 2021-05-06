
library(torchdrnn)

model <- nn_drnn(input_size = 10, hidden_size = 10,
                 num_layers = 4, dropout = 0,
                 cell_type = "gru")

x <- torch_randn(10, 1, 10)
model(x)

test_that("reshape tensor works", {
  batch_size <- 24
  input_size <- 3 #is this input_size?
  dilation <- 2

  x <- torch_randn(batch_size, input_size, 10)
  dilated_x <- model$prepare_input(input = x, dilation = dilation)
  # Correct shape of dilated_x
  expected_shape <- c(batch_size / dilation, input_size * dilation, 10)
  expect_equal(dilated_x$shape, expected_shape)

  # Correct creation of dilated_x
  expected_last_dilated_x <- x[dilation:batch_size:dilation]
  last_dilated_x <- dilated_x[, (input_size + 1):dilated_x$size(1)]
  expect_equal(last_dilated_x, expected_last_dilated_x)

  #Correct undilation of dilated_x
  expected_x <- model$split_output(dilated_output = dilated_x,
                                   dilation = dilation)
  expect_equal(expected_x, x)
})
