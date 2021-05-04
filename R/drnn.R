pacman::p_load(torch)


t <- torch_randn(3, 3)

drnn <- nn_module(
  classname = "DRNN",

  initialize = function(input_size, hidden_size, num_layers,
                        dropout = 0, cell_type = "GRU",
                        batch_first = FALSE){
    self$dilations <- purrr::map_dbl(1:num_layers, ~2**(. - 1))
    self$cell_type <- cell_type
    self$batch_first <- batch_first

    layers <- list()
    cell <- switch (self$cell_type,
      "GRU" = nn_gru,
      "RNN" = nn_rnn,
      "LSTM" = nn_lstm,
      stop("Invalid `cell_type` value.")
    )

    for (i in 1:num_layers) {
      if (i == 1) {
        layers[[i]] <- cell(input_size = input_size,
                          hidden_size = hidden_size, dropout = dropout)
      } else {
        layers[[i]] <- cell(input_size = hidden_size,
                          hidden_size = hidden_size, dropout = dropout)
      }
    }

    self$cells <- do.call(nn_sequential, layers)
  },

  forward = function(input, hx = NULL){
    if (self$batch_first) {
      inputs <- inputs$transpose(0, 1)
    }
    outputs <- list()

x

  }

)
