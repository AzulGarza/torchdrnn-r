
library(torchdrnn)
library(zeallot)
library(testthat)


n_time <- 24
batch_size <- 3
n_feats <- 10 #input size
dilation <- 2
n_output_feats <- 20 #hidden size
num_layers <- 4

model <- nn_drnn(input_size = n_feats, hidden_size = n_output_feats,
                 num_layers = num_layers, cell_type = "gru")

test_that("forward correct shape", {
  x <- torch_rand(n_time, batch_size, n_feats)
  c(output, hidden) %<-% model(x)
  expect_equal(x$shape, c(n_time, batch_size, n_feats))
})


test_that("reshape tensor works", {
  dilation <- 2

  x <- torch_randn(n_time, batch_size, n_feats)
  c(input, dummy) %<-% model$pad_input(input = x, n_steps = n_time, dilation = dilation)
  dilated_x <- model$prepare_input(input = input, dilation = dilation)
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
  expect_equal(x, expected_x)
})

test_that("Correct shape hidden list", {
  x <- torch_randn(n_time, batch_size, n_feats)
  c(output, hidden) %<-% model(x)
  expect_equal(length(hidden), num_layers)

  expected_sizes <- list(
    c(1, batch_size * 2 ** 0, n_output_feats),
    c(1, batch_size * 2 ** 1, n_output_feats),
    c(1, batch_size * 2 ** 2, n_output_feats),
    c(1, batch_size * 2 ** 3, n_output_feats)
  )

  for(i in 1:num_layers){
    expect_equal(hidden[[i]]$shape, expected_sizes[[i]])
  }
})

test_that("forward with predefined hidden", {
  hidden <- vector(mode = "list", length = num_layers)
  for (i in 1:num_layers) {
    hidden[[i]] <- torch_rand(1, batch_size * 2 ** (i - 1), n_output_feats)
  }

  x <- torch_rand(n_time, batch_size, n_feats)
  c(output, hidden) %<-% model(x, hx = hidden)

})
