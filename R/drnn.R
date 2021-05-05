pacman::p_load(torch, zeallot)


t <- torch_randn(10, 10, 10)

nn_drnn <- nn_module(
  classname = "nn_drnn",

  initialize = function(input_size, hidden_size, num_layers,
                        dropout = 0, cell_type = "gru",
                        batch_first = FALSE){
    self$num_layers <- num_layers
    self$dilations <- purrr::map_dbl(1:num_layers, ~2**(.-1))
    self$cell_type <- cell_type
    self$batch_first <- batch_first

    cell <- switch (self$cell_type,
      "gru" = nn_gru,
      "rnn" = nn_rnn,
      "lstm" = nn_lstm,
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
    if (self$batch_first) input <- input$transpose(1, 2)

    output <- vector(mode = "list", length = self$num_layers)
    for (i in 1:self$num_layers) {
      cell <- self$cells[[i]]
      dilation <- self$dilations[[i]]

      if (is.null(hx)){
        input <- self$drnn_layer(cell, input, dilation)
      } else {
        hidden <- hx[[i]]
        c(input, hx[[i]]) %<-% self$drnn_layer(cell, input, dilation, hidden)
      }
      output[[i]] <- input[, -dilation:-1]

    }

    if (self$batch_first) input <- input$transpose(1, 2)

    list(input, output)
  },

  drnn_layer = function(cell, input, rate, hx = NULL){
    n_steps <- length(input)
    batch_size <- input$size(1)
    hidden_size <- cell$hidden_size

    c(input, dummy) %<-% self$pad_input(input, n_steps, rate)
    dilated_input <- self$prepare_input(input, rate)

    if (is.null(hx)) {
      c(dilated_output, hidden) <- self$apply_cell(dilated_input, cell, batch_size, rate, hidden_size)
    } else {
      hidden <- self$prepare_input(hidden, rate)
      c(dilated_output, hidden) <- self$apply_cell(dilated_input, cell, batch_size, rate, hidden_size, hidden)
    }

    splitted_output <- self$split_output(dilated_output, rate)
    output <- self$unpad_output(splitted_output, n_steps)

    list(output, hidden)
  },

  apply_cell = function(dilated_input, cell, batch_size, rate, hidden_size, hx = NULL){
    device <- dilated_input$device

    if (is.null(hx)) {
      c(c, m) %<-% self$init_hidden(batch_size * rate, hidden_size,
                                    device = device)

      if(self$cell_type == "LSTM"){
        hidden <- list(c$unsqueeze(1), m$unsqueeze(1))
      } else {
        hidden <- c$unsqueeze(1)
      }

    }

    c(dilated_output, hidden) %<-% cell(dilated_input, hidden)

    list(dilated_input, hidden)
  },

  unpad_output = function(splitted_output, n_steps){
    splitted_output[1:n_steps]
  },

  split_output = function(dilated_output, dilation){
    batch_size <- dilated_output$size(2) %/% dilation

    blocks <- vector(mode = 'list', length = dilation)
    for (i in 1:dilation) {
      first_idx <- 1 + (i - 1) * batch_size
      last_idx <- i * batch_size
      blocks[[i]] <- dilated_output[, first_idx:last_idx,]
    }

    interleaved <- torch_stack(blocks)$transpose(2, 1)$contiguous()
    interleaved <- interleaved$view(c(dilated_output$size(1) * dilation,
                                      batch_size,
                                      dilated_output$size(3)))

    interleaved
  },

  pad_input = function(input, n_steps, rate){
     is_even <- (n_steps %% rate) == 0

     if (!is_even) {
       dilated_steps <- n_steps %/% rate + 1
       zeros_ <- torch_zeros(dilated_steps * rate - input$size(1),
                             input$size(1), input$size(2),
                             dtype = input$dtype, device = input$device)
        input <- torch_cat(c(input, zeros_))
     } else {
       dilated_steps <- n_steps %/% rate
     }

     list(input, dilated_steps)
  },

  prepare_input = function(input, dilation){
    dilated_input <- vector(mode = "list", length = dilation)
    n_input <- input$size(1)
    for (i in 1:dilation) {
      dilated_input[[i]] <- input[i:n_input:dilation]
    }

    dilated_input <- torch_cat(dilated_input, dim = 2)

    dilated_input
  },

  init_hidden = function(batch_size, hidden_dim, device){
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


