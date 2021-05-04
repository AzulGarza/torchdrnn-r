pacman::p_load(torch)


t <- torch_randn(10, 10, 10)

drnn <- nn_module(
  classname = "drnn",

  initialize = function(input_size, hidden_size, num_layers,
                        dropout = 0, cell_type = "GRU",
                        batch_first = FALSE){
    self$num_layers <- num_layers
    self$dilations <- purrr::map_dbl(1:num_layers, ~2**(.-1))
    self$cell_type <- cell_type
    self$batch_first <- batch_first

    cell <- switch (self$cell_type,
      "GRU" = nn_gru,
      "RNN" = nn_rnn,
      "LSTM" = nn_lstm,
      stop("Invalid `cell_type` value.")
    )

    self$cells <- nn_module_list()
    for (i in 1:num_layers) {
      if (i == 1) {
        self$cells$append(cell(input_size = input_size,
                               hidden_size = hidden_size, dropout = dropout))
      } else {
        self$cells$append(cell(input_size = hidden_size,
                               hidden_size = hidden_size, dropout = dropout))
      }
    }
  },

  forward = function(input, hx = NULL){
    if (self$batch_first) inputs <- inputs$transpose(1, 2)

    outputs <- vector(mode = "list", length = self$num_layers)
    for (i in 1:self$num_layers) {
      cell <- self$cells[[i]]
      dilation <- self$dilations[[i]]

      if (is.null(hx)){
        inputs <- self$drnn_layer(cell, input, dilation)
      } else {
        hidden <- hx[[i]]
        c(inputs, hx[[i]]) %<-% self$drnn_layer(cell, input, dilation, hidden)
      }
      outputs[[i]] <- inputs[, -dilation:-1]

    }

    if (self$batch_first) inputs <- inputs$transpose(1, 2)

    list(inputs, outputs)
  },

  drnn_layer = function(cell, inputs, rate, hx = NULL){
    n_steps <- length(inputs)
    batch_size <- inputs[[1]]$size(1)
    hidden_size <- cell$hidden_size

    c(inputs, dummy) %<-% self$pad_inputs(inputs, n_steps, rate)
    dilated_inputs <- self$prepare_inputs(inputs, rate)

    if (is.null(hx)) {
      c(dilated_outputs, hidden) <- self$apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
    } else {
      hidden <- self$prepare_inputs(hidden, rate)
      c(dilated_outputs, hidden) <- self$apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden)
    }

    splitted_outputs <- self$split_outputs(dilated_outputs, rate)
    outputs <- self$unpad_outputs(splitted_outputs, n_steps)

    list(outputs, hidden)
  },

  apply_cell = function(dilated_inputs, cell, batch_size, rate, hidden_size, hx = NULL){
    device <- dilated_inputs$device

    if (is.null(hx)) {
      c(c, m) %<-% self$init_hidden(batch_size * rate, hidden_size,
                                    device = device)

      if(self$cell_type == "LSTM"){
        hidden <- list(c$unsqueeze(1), m$unsqueeze(1))
      } else {
        hidden <- c$unsqueeze(1)
      }

    }

    c(dilated_outputs, hidden) %<-% cell(dilated_inputs, hidden)

    list(dilated_inputs, hidden)
  },

  unpad_outputs = function(splitted_outputs, n_steps){
    splitted_outputs[1:n_steps]
  },

  split_outputs = function(dilated_outputs, rate){
    batch_size <- dilated_outputs$size(1) %/% rate

    blocks <- vector(mode = 'list', length = rate)
    for (i in 1:rate) {
      blocks[[i]] <- dilated_outputs[, (i - 1) * batch_size:i * batch_size,]
    }

    interleaved <- torch_stack(blocks)$transpose(2, 1)$contiguous()
    interleaved <- interleaved$view(dilated_outputs$size(1) * rate,
                                    batch_size,
                                    dilated_outputs$size(2))

    interleaved
  },

  pad_inputs = function(inputs, n_steps, rate){
     is_even <- (n_steps %% rate) == 0

     if (!is_even) {
       dilated_steps <- n_steps %/% rate + 1
       zeros_ <- torch_zeros(dilated_steps * rate - inputs$size(1),
                             inputs$size(1), inputs$size(2),
                             dtype = inputs$dtype, device = inputs$device)
        inputs <- torch_cat(c(inputs, zeros_))
     } else {
       dilated_steps <- n_steps %/% rate
     }

     list(inputs, dilated_steps)
  },

  prepare_inputs = function(inputs, rate){
    dilated_inputs <- vector(model = "list", length = rate)
    for (i in 1:rate) {
      n_input <- inputs$size(1)
      dilated_inputs[[i]] <- inputs[i:n_input:rate]
    }

    dilated_inputs <- torch_cat(dilated_inputs, dim = 2)

    dilated_inputs
  },

  init_hidden = function(batch_size, hidden_dim, dtype){
    hidden <- torch_zeros(batch_size, hidden_dim,
                          device = device)

    if (self$cell_type == "LSTM") {
      memory <- torch_zeros(batch_size, hidden_dim,
                            device = device)
    } else {
      memory <- NULL
    }

    list(hidden, memory)
  }

)
