
library(torchdrnn)
library(zeallot)
library(testthat)


n_time <- 24
batch_size <- 3
n_feats <- 10 #input size
dilation <- 2
n_output_feats <- 20 #hidden size

model <- nn_drnn(input_size = n_feats, hidden_size = n_output_feats,
                 num_layers = 4, dropout = 0,
                 cell_type = "gru")

test_that("forward correct shape", {
  x <- torch_rand(n_time, batch_size, n_feats)
  c(output, hidden) %<-% model(x)
  expect_equal(x$shape, c(n_time, batch_size, n_feats))
})


test_that("reshape tensor works", {
  dilation <- 2

  x <- torch_randn(n_time, batch_size, n_feats)
  dilated_x <- model$prepare_input(input = x, dilation = dilation)
  # Correct shape of dilated_x
  expected_shape <- c(n_time / dilation, batch_size * dilation, n_feats)
  expect_equal(dilated_x$shape, expected_shape)

  # Correct creation of dilated_x
  expected_last_dilated_x <- x[dilation:n_time:dilation]
  last_dilated_x <- dilated_x[, (n_feats + 1):dilated_x$size(1)]
  expect_equal(last_dilated_x, expected_last_dilated_x)

  #Correct undilation of dilated_x
  expected_x <- model$split_output(dilated_output = dilated_x,
                                   dilation = dilation)
  expect_equal(expected_x, x)
})

