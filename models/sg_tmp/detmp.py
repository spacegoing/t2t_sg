# Decoder
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              save_weights_to=save_weights_to,
              max_relative_position=hparams.max_relative_position,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims)
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams,
              conv_padding="LEFT", nonpadding_mask=nonpadding)
          x = common_layers.layer_postprocess(x, y, hparams)

