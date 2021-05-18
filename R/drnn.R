#' DilatedRNN module
#'
#' Applies a multi-layer Elman RNN with \eqn{\tanh} or \eqn{\mbox{ReLU}} non-linearity
#' to an input sequence.
#'
#' For each element in the input sequence, each layer computes the following
#' function:
#'
#' \deqn{
#' h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
#' }
#'
#' where \eqn{h_t} is the hidden state at time `t`, \eqn{x_t} is
#' the input at time `t`, and \eqn{h_{(t-1)}} is the hidden state of the
#' previous layer at time `t-1` or the initial hidden state at time `0`.
#' If `nonlinearity` is `'relu'`, then \eqn{\mbox{ReLU}} is used instead of
#' \eqn{\tanh}.
#'
#' @param input_size The number of expected features in the input `x`
#' @param hidden_size The number of features in the hidden state `h`
#' @param num_layers Number of recurrent layers. E.g., setting `num_layers=2`
#'   would mean stacking two RNNs together to form a `stacked RNN`,
#'   with the second RNN taking in outputs of the first RNN and
#'   computing the final results. Default: 1
#' @param nonlinearity The non-linearity to use. Can be either `'tanh'` or
#'   `'relu'`. Default: `'tanh'`
#' @param bias If `FALSE`, then the layer does not use bias weights `b_ih` and
#'   `b_hh`. Default: `TRUE`
#' @param batch_first If `TRUE`, then the input and output tensors are provided
#'   as `(batch, seq, feature)`. Default: `FALSE`
#' @param dropout If non-zero, introduces a `Dropout` layer on the outputs of each
#'   RNN layer except the last layer, with dropout probability equal to
#'   `dropout`. Default: 0
#' @param bidirectional If `TRUE`, becomes a bidirectional RNN. Default: `FALSE`
#' @param ... other arguments that can be passed to the super class.
#'
#' @section Inputs:
#'
#' - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
#' of the input sequence. The input can also be a packed variable length
#' sequence.
#' - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#' containing the initial hidden state for each element in the batch.
#' Defaults to zero if not provided. If the RNN is bidirectional,
#' num_directions should be 2, else it should be 1.
#'
#'
#'
#' @section Outputs:
#'
#' - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
#' containing the output features (`h_t`) from the last layer of the RNN,
#' for each `t`.  If a :class:`nn_packed_sequence` has
#' been given as the input, the output will also be a packed sequence.
#' For the unpacked case, the directions can be separated
#' using `output$view(seq_len, batch, num_directions, hidden_size)`,
#' with forward and backward being direction `0` and `1` respectively.
#' Similarly, the directions can be separated in the packed case.
#'
#' - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#' containing the hidden state for `t = seq_len`.
#' Like *output*, the layers can be separated using
#' `h_n$view(num_layers, num_directions, batch, hidden_size)`.
#'
#' @section Shape:
#'
#' - Input1: \eqn{(L, N, H_{in})} tensor containing input features where
#'  \eqn{H_{in}=\mbox{input\_size}} and `L` represents a sequence length.
#' - Input2: \eqn{(S, N, H_{out})} tensor
#'   containing the initial hidden state for each element in the batch.
#'   \eqn{H_{out}=\mbox{hidden\_size}}
#'   Defaults to zero if not provided. where \eqn{S=\mbox{num\_layers} * \mbox{num\_directions}}
#'   If the RNN is bidirectional, num_directions should be 2, else it should be 1.
#' - Output1: \eqn{(L, N, H_{all})} where \eqn{H_{all}=\mbox{num\_directions} * \mbox{hidden\_size}}
#' - Output2: \eqn{(S, N, H_{out})} tensor containing the next hidden state
#'   for each element in the batch
#'
#' @section Attributes:
#' - `weight_ih_l[k]`: the learnable input-hidden weights of the k-th layer,
#'   of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
#'   `(hidden_size, num_directions * hidden_size)`
#' - `weight_hh_l[k]`: the learnable hidden-hidden weights of the k-th layer,
#'   of shape `(hidden_size, hidden_size)`
#' - `bias_ih_l[k]`: the learnable input-hidden bias of the k-th layer,
#'   of shape `(hidden_size)`
#' - `bias_hh_l[k]`: the learnable hidden-hidden bias of the k-th layer,
#'   of shape `(hidden_size)`
#'
#' @section Note:
#'
#' All the weights and biases are initialized from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})}
#' where \eqn{k = \frac{1}{\mbox{hidden\_size}}}
#'
#' @examples
#' drnn <- nn_drnn(10, 20, 2)
#' input <- torch_randn(5, 3, 10)
#' h0 <- torch_randn(2, 3, 20)
#' rnn(input, h0)
#'
#' @export
nn_drnn <- nn_module(
  classname = "nn_drnn",

  initialize = function(input_size, hidden_size, num_layers,
                        cell_type = "gru", dropout = 0,
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

