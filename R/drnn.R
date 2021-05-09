pacman::p_load(torch, zeallot)


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

    if (is.null(hx)) hx <- vector(mode = "list", length = self$num_layers)
    output <- input
    for (i in 1:self$num_layers) {
      cell <- self$cells[[i]]
      dilation <- self$dilations[[i]]
      c(output, hx[[i]]) %<-% self$drnn_layer(cell, output, dilation, hx[[i]])

    }

    if (self$batch_first) output <- output$transpose(1, 2)
    #hx <- torch_cat(hx)

    list(output, hx)
  },

  drnn_layer = function(cell, input, dilation, hx = NULL){
    c(n_steps, batch_size) %<-% input$size(1:2)
    hidden_size <- cell$hidden_size

    c(input, dummy) %<-% self$pad_input(input, n_steps, dilation)
    dilated_input <- self$prepare_input(input, dilation)

    c(dilated_output, hx) %<-% self$apply_cell(dilated_input=dilated_input,
                                               cell=cell,
                                               batch_size=batch_size,
                                               dilation=dilation,
                                               hidden_size=hidden_size,
                                               hx=hx)

    splitted_output <- self$split_output(dilated_output, dilation)
    output <- self$unpad_output(splitted_output, n_steps)

    list(output, hx)
  },

  apply_cell = function(dilated_input, cell, batch_size, dilation, hidden_size, hx = NULL){
    device <- dilated_input$device

    if (is.null(hx)) {
      hx <- torch_zeros(batch_size * dilation,
                        hidden_size,
                        dtype = dilated_input$dtype,
                        device = dilated_input$device)
      hx <- hx$unsqueeze(1)

      if(self$cell_type == "LSTM"){
        hx <- list(hx, hx)
      }
    }
    c(dilated_output, hx) %<-% cell(dilated_input, hx)

    list(dilated_output, hx)
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

  pad_input = function(input, n_steps, dilation){
     is_even <- (n_steps %% dilation) == 0

     if (!is_even) {
       dilated_steps <- n_steps %/% dilation + 1
       zeros_ <- torch_zeros(dilated_steps * dilation - input$size(1),
                             input$size(2), input$size(3),
                             dtype = input$dtype, device = input$device)
       input <- torch_cat(c(input, zeros_))
     } else {
       dilated_steps <- n_steps %/% dilation
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
  }
)

