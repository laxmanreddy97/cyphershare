import tensorflow as tf
import numpy as np

# 1. Load the pre-trained MobileNet V1
base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=True)

# ==========================================
# 2. Create the Client Model (Encoder Head)
# ==========================================
# UPDATED: Cut at conv_dw_6_relu to include depthwise spatial filtering
client_layer = base_model.get_layer('conv_dw_6_relu')
client_model = tf.keras.Model(inputs=base_model.input, outputs=client_layer.output)

# Save Client Model to TFLite
client_converter = tf.lite.TFLiteConverter.from_keras_model(client_model)
with open('encoder_head.tflite', 'wb') as f:
    f.write(client_converter.convert())

print("Client model saved.")

# ==========================================
# 3. Perturbation & Information Score Analysis
# ==========================================
target_layer = base_model.get_layer('conv_pw_6')
layer_weights = target_layer.get_weights()
# Safely extract weights (ignoring biases since MobileNet pointwise layers omit them)
weights = layer_weights[0] 

# Calculate information score proxy (L1 norm of weights per filter)
filter_scores = np.sum(np.abs(weights), axis=(0, 1, 2))
sorted_indices = np.argsort(filter_scores)[::-1]

# Designate the top 15% as critical (Trusted Owner), 85% as bulk (Third-Party)
split_idx = int(len(sorted_indices) * 0.15)
critical_indices = list(sorted_indices[:split_idx])
bulk_indices = list(sorted_indices[split_idx:])

# TRACK THE SHUFFLE: The tensor will be ordered exactly like this after concatenation
concat_indices = critical_indices + bulk_indices

# Create an inverse mapping array to unscramble the channels back to 0-511
inverse_indices = np.argsort(concat_indices).astype(np.int32)

print(f"Routing {len(critical_indices)} critical neurons to Trusted Owner.")
print(f"Routing {len(bulk_indices)} bulk neurons to Third Party.")

# ==========================================
# 4. Trusted Owner Model (Critical Neurons)
# ==========================================
owner_input = tf.keras.Input(shape=client_layer.output.shape[1:])
owner_conv = tf.keras.layers.Conv2D(
    filters=len(critical_indices),
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    use_bias=False,
    name='trusted_pw_6'
)(owner_input)
owner_model = tf.keras.Model(inputs=owner_input, outputs=owner_conv)

# Inject only the weights array
owner_model.get_layer('trusted_pw_6').set_weights([
    weights[:, :, :, critical_indices]
])
owner_model.save('trusted_owner.h5')

# ==========================================
# 5. Third-Party Model (Bulk + Remainder of Network)
# ==========================================
tp_client_input = tf.keras.Input(shape=client_layer.output.shape[1:], name="client_latent")
tp_owner_input = tf.keras.Input(shape=owner_conv.shape[1:], name="owner_processed")
# NEW: Input for the noise cancellation offset from the Trusted Owner
tp_noise_offset = tf.keras.Input(shape=(14, 14, len(bulk_indices)), name="bulk_noise_offset")

tp_bulk_conv = tf.keras.layers.Conv2D(
    filters=len(bulk_indices),
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    use_bias=False,
    name='bulk_pw_6'
)(tp_client_input)

# NEW: Subtract the noise offset to recover the clean bulk tensor natively in the graph
clean_bulk_conv = tf.keras.layers.Subtract(name="noise_cancellation")([tp_bulk_conv, tp_noise_offset])

tp_bulk_model_temp = tf.keras.Model(inputs=tp_client_input, outputs=tp_bulk_conv)
tp_bulk_model_temp.get_layer('bulk_pw_6').set_weights([
    weights[:, :, :, bulk_indices]
])

# Use the CLEANED bulk tensor for recombination
recombined = tf.keras.layers.Concatenate(axis=-1)([tp_owner_input, clean_bulk_conv])

# ... (Keep Section 6 exactly the same, but update the final model definition below) ...


# ==========================================
# 6. RESTORE ORIGINAL CHANNEL ORDER (NATIVE FIX)
# ==========================================
# Use a native 1x1 Conv2D layer to physically route the channels
total_channels = len(concat_indices)
tp_reorder_conv = tf.keras.layers.Conv2D(
    filters=total_channels,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    use_bias=False,
    trainable=False,
    name='channel_reorder_conv'
)(recombined)

# ==========================================
# 7. APPLY BATCH NORMALIZATION & REST OF NETWORK
# ==========================================
# Apply the original Batch Normalization layer to the restored tensor
bn_layer = base_model.get_layer('conv_pw_6_bn')
x = bn_layer(tp_reorder_conv)

# Continue the loop starting FROM the ReLU layer
start_index = base_model.layers.index(base_model.get_layer('conv_pw_6_relu'))

for layer in base_model.layers[start_index:]:
    x = layer(x)

# Update the final model definition in Section 7 to accept all three inputs
third_party_model = tf.keras.Model(
    inputs=[tp_client_input, tp_owner_input, tp_noise_offset], 
    outputs=x
)

# Build the permutation matrix weights to unscramble the channels
# Shape: (kernel_h, kernel_w, in_channels, out_channels)
permute_weights = np.zeros((1, 1, total_channels, total_channels), dtype=np.float32)
for out_channel in range(total_channels):
    # Route the scrambled input channel to its proper output position
    in_channel = inverse_indices[out_channel]
    permute_weights[0, 0, in_channel, out_channel] = 1.0

# Inject the routing weights into our layer
third_party_model.get_layer('channel_reorder_conv').set_weights([permute_weights])

third_party_model.save('third_party_bulk.h5')
print("3-Tier Split Complete. Native channel alignment, BN restored, and downsampled to 14x14.")